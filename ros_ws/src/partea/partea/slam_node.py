import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseArray, Quaternion, Pose
from sensor_msgs.msg import LaserScan
from custom_msgs.msg import Belief
from scipy.spatial.transform import Rotation as R

from algoritmos import likelihood_field as lf
from algoritmos import motion_model as mm
from algoritmos import particle_filter as pf
from algoritmos import occupancy_grid as og
from algoritmos import scan_matching as sm


def extract_pose(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    q_w = msg.pose.pose.orientation.w
    q_x = msg.pose.pose.orientation.x
    q_y = msg.pose.pose.orientation.y
    q_z = msg.pose.pose.orientation.z

    current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
    theta = current_rotation.as_euler('xyz', degrees=False)[2]

    return x, y, theta


def quaternion_from_yaw(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = np.sin(yaw / 2.0)
    q.w = np.cos(yaw / 2.0)
    return q


def compute_deltas(pose:tuple, last_odom:tuple) -> dict[str, float]:
    x, y, theta = pose
    # Compute translation difference
    dx = x - last_odom[0]
    dy = y - last_odom[1]
    delta_t = np.sqrt(dx**2 + dy**2)


    # Umbral en METROS: por debajo de esto la traslacion es ruido y atan2(dy,dx) daria
    # un delta_rot1 basura. En rotaciones (casi) en el lugar se trata como giro puro.
    if delta_t > 0.01:
        delta_rot1 = np.arctan2(dy, dx) - last_odom[2]
        delta_rot2 = theta - last_odom[2] - delta_rot1
    else:
        # Sin traslacion significativa → rotacion en el lugar (todo el giro va en delta_rot2)
        delta_rot1 = 0.0
        delta_rot2 = theta - last_odom[2]

    # Normalize angles
    delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
    delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))

    return {'t':delta_t, 'r1':delta_rot1, 'r2':delta_rot2}


def pose_to_matrix(pose):
    # Pose 2D (x, y, theta) -> matriz homogenea 3x3
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]])


def matrix_to_pose(T):
    # Matriz homogenea 3x3 -> pose 2D (x, y, theta)
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def lidar_offset_from_rTlidar(rTlidar):
    # Extrae la transformacion 2D (x, y, yaw) del 4x4 homogeneo base->lidar
    x, y = rTlidar[0, 3], rTlidar[1, 3]
    yaw = np.arctan2(rTlidar[1, 0], rTlidar[0, 0])
    return np.array([x, y, yaw])


def robot_to_sensor(robot_pose, lidar_offset):
    # Pose del robot (base) -> pose del lidar: world_T_sensor = world_T_robot @ base_T_lidar
    return matrix_to_pose(pose_to_matrix(robot_pose) @ pose_to_matrix(lidar_offset))


def sensor_to_robot(sensor_pose, lidar_offset):
    # Inversa: world_T_robot = world_T_sensor @ inv(base_T_lidar)
    return matrix_to_pose(pose_to_matrix(sensor_pose) @ np.linalg.inv(pose_to_matrix(lidar_offset)))


class SLAM(Node):
    def __init__(self,
                 N:int=15,
                 odom_noise:np.array=np.array([.1, .1, .1, .1]),
                 map_size:float=4.0,
                 map_resolution:float=0.02,
                 likelihood_field_sigma:float = 0.1,
                 update_k:int=3,
                 scan_matching_subsample:int=16,
                 scan_matching_noise:np.array=np.array([0.01, 0.01, 0.005]),
                 iters_to_recompute_lf:int=3):
        super().__init__('slam')

        self.declare_parameter("robot", "simulado")
        robot = self.get_parameter("robot").get_parameter_value().string_value

        if robot not in ("real", "simulado"):
            self.get_logger().warn(f"Valor inválido para 'robot': '{robot}'. Usando 'simulado'.")
            robot = "simulado"

        if robot == "simulado":
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
            self.calodom_sub = self.create_subscription(Odometry, "/calc_odom", self.odom_callback, 10)
            self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

            self.belief_pub = self.create_publisher(Belief, '/belief', 10)
            self.map_pub = self.create_publisher(OccupancyGrid, '/map', qos)
            self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)

            # En el simulador el lidar coincide con la base: transformacion identidad
            self.rTlidar = np.eye(4)
        else:
            # Los sensores del TB publican BEST_EFFORT: el subscriber debe matchear o no recibe nada
            sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            # El mapa conviene latcheado para que RViz lo reciba aunque se conecte tarde
            map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
            self.calcodom_sub = self.create_subscription(Odometry, "/tb4_1/odom", self.odom_callback, sensor_qos)
            self.scan_sub = self.create_subscription(LaserScan, "/tb4_1/scan", self.scan_callback, sensor_qos)

            self.belief_pub = self.create_publisher(Belief, '/tb4_1/belief', 10)
            self.map_pub = self.create_publisher(OccupancyGrid, '/tb4_1/map', map_qos)
            self.particles_pub = self.create_publisher(PoseArray, '/tb4_1/particles', 10)

            self.rTlidar = np.array([[0.0, -1.0,  0.0, -0.04],
                                     [1.0,  0.0,  0.0,   0.0],
                                     [0.0,  0.0,  1.0, 0.193],
                                     [0.0,  0.0,  0.0,   1.0]])

        # Transformacion 2D (x, y, yaw) base->lidar, derivada de rTlidar. Se usa para
        # proyectar los rayos desde la pose real del lidar y no desde el centro del robot.
        self.lidar_offset = lidar_offset_from_rTlidar(self.rTlidar)

        # El TB4 marca retornos invalidos con intensity == 0; el sim no publica intensidades
        self.filter_intensity = (robot == "real")

        self.N = N                      # cantidad de particulas
        self.map_update_k = update_k    # solo se actualiza el mapa de las K partículas con mayor peso

        self.noise = odom_noise
        self.sigma = likelihood_field_sigma

        self.last_odom = None   # ultima lectura de odometria (se actualiza en cada msg)
        self.prev_odom = None   # odometria usada en el scan anterior (para el delta acumulado)
        self.odom_frame = "odom"  # frame en el que se publican mapa y particulas; se toma del msg de odom
       
        map_size_m = map_size           # lado del mapa [m]
        resolution = map_resolution     # lado de la celda [m]
        width  = int(map_size_m / resolution)
        height = int(map_size_m / resolution)
        origin_x = -map_size_m / 2
        origin_y = -map_size_m / 2

        # Inicialización de partículas, pesos y mapas
        self.particles = np.zeros((self.N, 3))      # cada fila es [x, y, theta]
        self.weights = np.ones(self.N) / self.N     # inicialización con pesos uniformes
        self.maps = [og.ParticleMap(width, height, resolution, origin_x, origin_y) for _ in range(self.N)]

        # Scan matching
        self.match_subsample = scan_matching_subsample  # subsample de rayos
        self.match_noise = scan_matching_noise          # jitter alrededor del optimo, para mantener diversidad
        self.map_subsample = 2  # subsample de rayos para construir el mapa
        
        self.scan_count = 0
        self.cached_lf = None
        self.lf_recompute_interval = iters_to_recompute_lf

        # Autoguardado del mapa: cada 'map_save_interval' scans se guarda el mapa de la
        # mejor particula en .npz, con EXACTAMENTE el mismo formato que save_map.py.
        # 'map_save_path' es la ruta base (sin extension); el archivo queda en el
        # directorio de trabajo del nodo (igual que save_map.py). interval<=0 lo desactiva.
        self.declare_parameter("map_save_path", "mapa_slam")
        self.declare_parameter("map_save_interval", 100)
        self.map_save_path = self.get_parameter("map_save_path").get_parameter_value().string_value
        self.map_save_interval = self.get_parameter("map_save_interval").get_parameter_value().integer_value
        if self.map_save_interval > 0:
            self.get_logger().info(
                f"Autoguardado de mapa cada {self.map_save_interval} scans en '{self.map_save_path}.npz'")

   
    def scan_callback(self, msg: LaserScan):
        # Actualizar solo el mapa de la mejor partícula (o las top-K) en vez de todas las N. Esto es una aproximación común de FastSLAM que reduce el cómputo drásticamente.
        # Alternativamente, actualizar todos pero con menos frecuencia.
        ranges = np.array(msg.ranges)

        # TB4: los retornos con intensity == 0 son invalidos -> los marcamos inf para que
        # el filtro isfinite (en update, compute_likelihood y make_beam_points) los descarte.
        if self.filter_intensity:
            intensities = np.asarray(msg.intensities)
            if intensities.size == ranges.size:
                ranges[intensities <= 0.0] = np.inf

        # Descartar lecturas de rango maximo (no-lecturas): no son obstaculos ni info
        # confiable. En inf, el isfinite las excluye del mapeo, el matching y el pesado.
        ranges[ranges >= msg.range_max] = np.inf
        ranges(ranges <= msg.range_min) = np.inf  # descartar tambien rangos invalidos por debajo del minimo

        scan_angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        scan_data = (msg.angle_min, msg.angle_increment, msg.range_max, ranges, msg.range_min, msg.intensities)


        # Prediccion: aplicar el motion model UNA sola vez por scan, con el delta de
        # odometria acumulado desde el scan anterior. Asi delta_rot1 se calcula sobre un
        # desplazamiento real (bien condicionado) y el ruido se agrega una vez, no por
        # cada mensaje de odom (~30 Hz), que es lo que dispersaba/corria la nube.
        if self.last_odom is not None:
            if self.prev_odom is not None:
                deltas = compute_deltas(self.last_odom, self.prev_odom)
                self.particles = mm.sample_motion_model_odometry(self.particles, deltas, self.noise)
            self.prev_odom = self.last_odom


        # Scan matching + pesado: cada particula contra SU PROPIO mapa (FastSLAM).
        # El field de cada particula refleja el mapa que ella misma construyo, asi el
        # peso mide que tan bien calza el scan con ESE mapa -> los pesos discriminan
        # entre buenos y malos mapas, y el resampling concentra en los buenos.
        bx, by = sm.make_beam_points(ranges, scan_angles, msg.range_min, self.match_subsample)

        for i in range(self.N):
            # Field del mapa de esta particula (hasta el scan anterior)
            field_i = lf.precompute_likelihood_field(self.maps[i], self.sigma)

            # Los rayos salen del lidar, no del centro del robot: se proyecta desde la
            # pose del sensor (base (+) rTlidar) y se vuelve a la del robot.
            sensor_pose = robot_to_sensor(self.particles[i], self.lidar_offset)
            matched_sensor = sm.scan_match(sensor_pose, bx, by, field_i)
            self.particles[i] = sensor_to_robot(matched_sensor, self.lidar_offset) + np.random.normal(0, self.match_noise)

            # Pesar con la pose refinada, contra el MISMO field (su propio mapa)
            sensor_pose = robot_to_sensor(self.particles[i], self.lidar_offset)
            w = lf.compute_likelihood(sensor_pose, ranges, scan_angles, msg.range_min, field_i)
            self.weights[i] *= w

        # Normalizar pesos
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights = np.ones(self.N) / self.N


        # Actualizar el mapa de CADA particula con su pose refinada. Ahora que el peso
        # depende del mapa propio, todas las particulas deben mantener su mapa al dia
        # (ya no vale el atajo de mapear solo las top-K).
        for i in range(self.N):
            self.maps[i].update(robot_to_sensor(self.particles[i], self.lidar_offset), scan_data, subsample=self.map_subsample)


        # Resamplear si N_eff bajo
        n_eff = pf.effective_sample_size(self.weights)
        if n_eff < self.N*0.5:
            self.particles, self.weights, self.maps = pf.sus(self.particles, self.weights, self.maps)


        # Publicar
        self.publish_results()

        self.scan_count += 1

        # Checkpoint periodico del mapa a .npz (mismo formato que save_map.py)
        if self.map_save_interval > 0 and self.scan_count % self.map_save_interval == 0:
            self.save_map_npz()


    def odom_callback(self, msg: Odometry):
        # Solo guarda la ultima odometria. La prediccion se hace una vez por scan
        # en scan_callback, con el delta acumulado desde el scan anterior.
        self.last_odom = extract_pose(msg)
        # La estimacion vive en el frame de la odometria que se integra (calc_odom en
        # sim, tb4_1/odom en el TB): publicamos mapa y particulas en ese frame.
        self.odom_frame = msg.header.frame_id

    def publish_particles(self, stamp):
        # publicar particulas
        particles = PoseArray()
        particles.header.frame_id = self.odom_frame  # mismo frame que el mapa: ambos son la estimacion SLAM
        particles.header.stamp = stamp
        poses = []
        for particle in self.particles:
            pose = Pose()
            pose.position.x = float(particle[0])
            pose.position.y = float(particle[1])
            pose.orientation = quaternion_from_yaw(particle[2])
            poses.append(pose)
        particles.poses = poses
        self.particles_pub.publish(particles)

    def publish_belief(self, best_particle):
        belief = Belief()
        belief.mu.x = float(best_particle[0])
        belief.mu.y = float(best_particle[1])
        belief.mu.theta = float(best_particle[2])
        belief.covariance = [0.0] * 9
        self.belief_pub.publish(belief)

    def publish_map(self, best_idx, stamp):
        best_map = self.maps[best_idx].to_occupancy_grid_msg(frame_id=self.odom_frame, stamp=stamp)
        self.map_pub.publish(best_map)

    def save_map_npz(self):
        # Guarda el mapa de la mejor particula en .npz con EXACTAMENTE el mismo formato
        # que save_map.py: se parte del OccupancyGrid (mapa publicado) y se extraen grid,
        # resolucion y origen de la misma manera, con las mismas claves del np.savez.
        best_idx = int(np.argmax(self.weights))
        msg = self.maps[best_idx].to_occupancy_grid_msg(
            frame_id=self.odom_frame, stamp=self.get_clock().now().to_msg())

        w, h = msg.info.width, msg.info.height
        # data es row-major desde la esquina inferior-izquierda (fila 0 = abajo)
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)

        res = msg.info.resolution
        o = msg.info.origin.position

        # .npz: grilla + metadata (resolucion y origen). Sin el origen, quien cargue el
        # mapa no sabe donde esta apoyada la grilla.
        np.savez(f"{self.map_save_path}.npz",
                 grid=grid,
                 resolution=res,
                 origin_x=o.x,
                 origin_y=o.y)

        self.get_logger().info(
            f"[checkpoint scan {self.scan_count}] Guardado {self.map_save_path}.npz  |  "
            f"{h}x{w} celdas, resolucion={res} m, origin=({o.x:.3f}, {o.y:.3f})")

    def publish_results(self):
        stamp = self.get_clock().now().to_msg()

        self.publish_particles(stamp)

        best_idx = np.argmax(self.weights)
        best_particle = self.particles[best_idx]

        self.publish_belief(best_particle)

        self.publish_map(best_idx, stamp)
 

def main(args=None):
    rclpy.init(args=args)
    node = SLAM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()