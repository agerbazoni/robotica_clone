import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Twist, PoseArray, Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from algoritmos import likelihood_field_B, motion_model, particle_filter, path_planning, path_following
from parteb.map_loader import MapData

# ------------------------------------------------------
# ESTADOS
# ------------------------------------------------------
IDLE = 0
WAITING_GOAL = 1
PLANNING = 2
NAVIGATING = 3
ADJUSTING_ANGLE = 4

def pose_to_matrix(pose):
    """
    Convierte una pose 2D (x, y, theta) en una matriz de transformación homogénea 3x3.
    Parámetros:
    - pose: contiene la posición (x, y) y orientación (theta) del robot.
    Retorna:
    - Una matriz de transformación homogénea 3x3 que representa la pose en el espacio 2D.    
    """
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]])

def matrix_to_pose(T):
    """
    Convierte una matriz de transformación homogénea 3x3 en una pose 2D (x, y, theta).
    Parámetros:
    - T: matriz de transformación homogénea 3x3.
    Retorna:
    - Una lista que contiene la posición (x, y) y orientación (theta) del robot.
    """
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])

def lidar_offset_from_rTlidar(rTlidar):
    """
    Calcula el offset del LIDAR a partir de la matriz de transformación rTlidar.
    Parámetros:
    - rTlidar: matriz de transformación 4x4 que representa la posición y orientación del LIDAR con respecto al robot.    
    Retorna:
    - Un array que contiene el offset del LIDAR en el espacio 2D (x, y, theta).
    """
    x, y = rTlidar[0, 3], rTlidar[1, 3]
    yaw = np.arctan2(rTlidar[1, 0], rTlidar[0, 0])
    return np.array([x, y, yaw])

def robot_to_sensor(robot_pose, lidar_offset):
    """
    Calcula la pose del sensor (LIDAR) a partir de la pose del robot y el offset del LIDAR.
    Parámetros:
    - robot_pose: contiene la posición (x, y) y orientación (theta) del robot.
    - lidar_offset: array que contiene el offset del LIDAR en el espacio 2D (x, y, theta).
    Retorna:
    - Una lista que contiene la posición (x, y) y orientación (theta) del sensor (LIDAR) en el espacio 2D.
    """
    return matrix_to_pose(pose_to_matrix(robot_pose) @ pose_to_matrix(lidar_offset))

class nodo_b(Node):
    """
    Nodo principal que implementa un sistema de navegación.
    """
    def __init__(self):
        super().__init__('nodo_b')

        self.noise = [0.01, 0.01, 0.01, 0.01]
        self.sigma = 0.2

        self.declare_parameter("map_path", "mapa.npz")
        map_path = self.get_parameter("map_path").get_parameter_value().string_value

        self.map_data = MapData(map_path)
        self.likelihood_field = likelihood_field_B.precompute_likelihood_field(self.map_data, self.sigma)

        self.state = IDLE

        # Inicialización de variables para el filtro de partículas
        self.N = 100
        self.particles = None
        self.weights = None
        
        # Inicialización de variable para la odometría
        self.last_odom = None
        
        # Inicialización de variables para la posición objetivo, la posición inicial y la pose estimada
        self.goal = None
        self.goal_dif = False
        self.inipos = None
        self.inipos_dif = False
        self.estimated_pose = None

        # Inicialización de variables para la planificación de rutas
        self.path = None
        self.radio_robot = 0.105
        self.margen = 0.06
        self.radius_cells = int((self.radio_robot + self.margen) / self.map_data.resolution)
        self.inflated = path_planning.inflate_map(self.map_data.occupied, self.radius_cells)

        # Inicialización para pure pursuit
        self.lookahead = 0.3

        # Inicialización del mapa dinámico y variables relacionadas con la detección de obstáculos
        self.dynamic_map = np.zeros((self.map_data.height, self.map_data.width), dtype=np.float32)
        self.dynamic_changed = False
        self.replan_attempts = 0
        self.obstacle_cooldown = 0
        self.current_omega = 0.0
        self.dynamic_update_counter = 0
        self.last_scan = None

        self.timer = self.create_timer(0.1, self.loop)

        # Configuración de parámetros para el TB3 (simulado) o TB4 (real)
        self.declare_parameter("robot", "simulado")
        robot = self.get_parameter("robot").get_parameter_value().string_value

        if robot not in ("real", "simulado"):
            self.get_logger().warn(f"Valor inválido para 'robot': '{robot}'. Usando 'simulado'.")
            robot = "simulado"

        if robot == "simulado":
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

            # Publishers
            self.publisher_vel = self.create_publisher(Twist, "/cmd_vel", 10)
            self.publish_estimated_pose = self.create_publisher(PoseStamped, '/estimated_pose', 10)
            self.publish_path = self.create_publisher(Path, '/planned_path', qos)
            self.publish_inflated_map = self.create_publisher(OccupancyGrid, '/inflated_map', qos)
            self.publisher_particles = self.create_publisher(PoseArray, '/particles', 10)
            self.publish_dynamic_map_pub = self.create_publisher(OccupancyGrid, '/dynamic_map', qos)

            # Subscribers
            self.subscribe_calcodom = self.create_subscription(Odometry, '/calc_odom', self.odom_callback, 10)
            self.subscribe_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
            self.subscribe_inipos = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.inipos_callback, 10)
            self.subscribe_goalpos = self.create_subscription(PoseStamped, '/goal_pose', self.goalpos_callback, 10)

            self.rTlidar = np.eye(4)
            self.rotation_error = 0.0
        else:
            # Los sensores del TB publican BEST_EFFORT
            sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            # Mapas/paths latcheados para que RViz los reciba aunque se conecte tarde
            map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

            # Publishers
            self.publisher_vel = self.create_publisher(Twist, "/tb4_0/cmd_vel", 10)
            self.publish_estimated_pose = self.create_publisher(PoseStamped, '/tb4_0/estimated_pose', 10)
            self.publish_path = self.create_publisher(Path, '/tb4_0/planned_path', map_qos)
            self.publish_inflated_map = self.create_publisher(OccupancyGrid, '/tb4_0/inflated_map', map_qos)
            self.publisher_particles = self.create_publisher(PoseArray, '/tb4_0/particles', 10)
            self.publish_dynamic_map_pub = self.create_publisher(OccupancyGrid, '/tb4_0/dynamic_map', map_qos)

            # Subscribers
            self.subscribe_calcodom = self.create_subscription(Odometry, '/tb4_0/odom', self.odom_callback, sensor_qos)
            self.subscribe_scan = self.create_subscription(LaserScan, '/tb4_0/scan', self.scan_callback, sensor_qos)
            self.subscribe_inipos = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.inipos_callback, 10)
            self.subscribe_goalpos = self.create_subscription(PoseStamped, '/goal_pose', self.goalpos_callback, 10)

            # LIDAR del TB4 rotado respecto al robot
            self.rTlidar = np.array([[0.0, -1.0,  0.0, -0.04],
                                     [1.0,  0.0,  0.0,   0.0],
                                     [0.0,  0.0,  1.0, 0.193],
                                     [0.0,  0.0,  0.0,   1.0]])
            # 20.5 grados inercia al girar a 1 rad/s
            self.rotation_error = np.deg2rad(20.5)

        self.robot = robot
        # Transformacion para proyectar los rayos desde la pose del lidar y no desde el centro del robot.
        self.lidar_offset = lidar_offset_from_rTlidar(self.rTlidar)

        self.publish_inflated(self.inflated)

# ------------------------------------------------------
# ODOM CALLBACK
# ------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        """
        Función callback para el tópico de odometría. Actualiza las partículas basándose en la información de odometría.
        Parámetros:
        - msg: mensaje de tipo Odometry.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q_w = msg.pose.pose.orientation.w
        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z

        current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
        theta = current_rotation.as_euler('xyz', degrees=False)[2]

        deltas = {'t': 0, 'r1': 0, 'r2': 0}

        if self.last_odom != None:
            dx = x - self.last_odom[0]
            dy = y - self.last_odom[1]
            delta_t = np.sqrt(dx**2 + dy**2)

            if delta_t > 1e-6:
                delta_rot1 = np.arctan2(dy, dx) - self.last_odom[2]
                delta_rot2 = theta - self.last_odom[2] - delta_rot1
            else:
                delta_rot1 = 0.0
                delta_rot2 = theta - self.last_odom[2]

            delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
            delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))

            if delta_t > 0.001 or abs(delta_rot1) > 0.01 or abs(delta_rot2) > 0.01:
                deltas['t'] = delta_t
                deltas['r1'] = delta_rot1
                deltas['r2'] = delta_rot2
                self.particles = motion_model.sample_motion_model_odometry(self.particles, deltas, self.noise)

        self.last_odom = (x, y, theta)

# ------------------------------------------------------
# SCAN CALLBACK
# ------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        """
        Función callback para el tópico del LIDAR. Actualiza los pesos de las partículas basándose en la información del escaneo.
        Parámetros:
        - msg: mensaje de tipo LaserScan.
        """
        self.last_scan = msg

        if self.particles is None:
            return

        ranges = np.array(msg.ranges, dtype=float)
        scan_angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # TB4: lecturas con intensidad 0 son invalidas, hay que descartarlas.
        # Se marcan como NaN para que los algoritmos las filtren con isfinite().
        if self.robot == "real" and len(msg.intensities) == len(ranges):
            intensities = np.array(msg.intensities, dtype=float)
            ranges[intensities == 0] = np.nan

        for i in range(self.N):
            # Los rayos salen del lidar, no del centro del robot: se proyecta desde la pose del sensor
            sensor_pose = robot_to_sensor(self.particles[i], self.lidar_offset)
            w = likelihood_field_B.compute_likelihood(sensor_pose, ranges, scan_angles, msg.range_min, self.likelihood_field, self.map_data)
            self.weights[i] *= w

        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.N) / self.N

        n_eff = particle_filter.effective_sample_size(self.weights)
        if n_eff < self.N / 1.5:
            self.particles, self.weights = particle_filter.sus(self.particles, self.weights, None)

        self.estimated_pose = particle_filter.get_selected_state(self.particles, self.weights)

# ------------------------------------------------------
# INIPOS CALLBACK
# ------------------------------------------------------
    def inipos_callback(self, msg: PoseWithCovarianceStamped):
        """
        Función callback para el tópico de posición inicial. Inicializa las partículas y actualiza la posición estimada.
        Parámetros:
        - msg: mensaje de tipo PoseWithCovarianceStamped.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q_w = msg.pose.pose.orientation.w
        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z

        current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
        theta = current_rotation.as_euler('xyz', degrees=False)[2]

        if self.inipos != [x, y, theta]:
            self.inipos = [x, y, theta]
            self.inipos_dif = True
            self.last_odom = None
            self.path = None
            self.publish_movement(0.0, 0.0)

            self.particles = np.column_stack([np.random.normal(x, 0.3, self.N),
                                              np.random.normal(y, 0.3, self.N),
                                              np.random.normal(theta, 0.2, self.N)])
            self.weights = np.ones(self.N) / self.N
            self.estimated_pose = [x, y, theta]

# ------------------------------------------------------
# GOALPOS CALLBACK
# ------------------------------------------------------
    def goalpos_callback(self, msg: PoseStamped):
        """
        Función callback para el tópico de posición objetivo. Actualiza la posición objetivo.
        Parámetros:
        - msg: mensaje de tipo PoseStamped.
        """
        x = msg.pose.position.x
        y = msg.pose.position.y

        q_x = msg.pose.orientation.x
        q_y = msg.pose.orientation.y
        q_z = msg.pose.orientation.z
        q_w = msg.pose.orientation.w

        current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
        theta = current_rotation.as_euler('xyz', degrees=False)[2]

        if self.goal != [x, y, theta]:
            self.goal = [x, y, theta]
            self.goal_dif = True
            self.path = None
            self.publish_movement(0.0, 0.0)

# ------------------------------------------------------
# PLAN ROUTE
# ------------------------------------------------------
    def plan_route(self):
        """
        Planea una ruta desde la posición actual hasta la posición objetivo utilizando el algoritmo A*.
        La ruta se almacena en self.path como una lista de coordenadas (x, y) en el mundo real. Si no se encuentra una ruta, self.path se establece en None.
        """
        dynamic_occupied = self.dynamic_map > 0.5
        dynamic_inflated = path_planning.inflate_map(dynamic_occupied, self.radius_cells) # Para un kernel circular para inflar el mapa y evitar obstáculos dinámicos y estáticos.
        combined = self.inflated | dynamic_inflated # Combina el mapa inflado estático con el mapa dinámico inflado para obtener un mapa combinado de ocupación.

        pose = particle_filter.get_selected_state(self.particles, self.weights)
        start = self.map_data.world_to_grid(pose[0], pose[1])
        goal = self.map_data.world_to_grid(self.goal[0], self.goal[1])

        sc, sr = start
        # Si el start está bloqueado, buscar celda libre más cercana
        if combined[sr, sc]:
            free_cells = np.argwhere(~combined)
            if len(free_cells) == 0:
                self.path = None
                return
            dists = np.hypot(free_cells[:, 0] - sr, free_cells[:, 1] - sc)
            nearest = free_cells[np.argmin(dists)]
            sr, sc = nearest[0], nearest[1]
            start = (sc, sr)

        path_cells = path_planning.a_star(combined, start, goal) # Calcula la ruta utilizando el algoritmo A* en el mapa combinado de ocupación.

        if path_cells is not None:
            self.path = [self.map_data.grid_to_world(c, r) for c, r in path_cells] # Convierte las coordenadas de la ruta de celdas del mapa a coordenadas del mundo real y almacena la ruta
        else:
            self.path = None

# ------------------------------------------------------
# OBSTACLE AVOIDANCE
# ------------------------------------------------------
    def update_dynamic_map(self):
        """
        Actualiza el mapa dinámico basado en la información del escaneo LIDAR y la posición estimada del robot.
        Marca las celdas ocupadas por obstáculos detectados y libera las celdas a lo largo de los rayos del LIDAR. 
        También aplica una decaída temporal a las celdas dinámicas para reflejar la desaparición de obstáculos con el tiempo.
        """
        if self.last_scan is None or self.estimated_pose is None:
            return

        ranges = np.array(self.last_scan.ranges, dtype=float)
        angles = self.last_scan.angle_min + np.arange(len(ranges)) * self.last_scan.angle_increment

        # TB4: descartar rayos con intensidad 0 (lecturas invalidas)
        if self.robot == "real" and len(self.last_scan.intensities) == len(ranges):
            intensities = np.asarray(self.last_scan.intensities, dtype=float)
            ranges[intensities == 0] = np.nan

        pose = robot_to_sensor(self.estimated_pose, self.lidar_offset)

        decay_rate = 0.02 # Hiperparámetro, cuanto más alto, más rápido desaparecen los obstáculos del mapa
        prev = (self.dynamic_map > 0.5)
        self.dynamic_map = np.clip(self.dynamic_map - decay_rate, 0.0, 1.0)

        for i, r in enumerate(ranges):
            if not np.isfinite(r) or r < self.last_scan.range_min:
                continue

            angle = pose[2] + angles[i]
            hit_obstacle = r <= self.last_scan.range_max # Determina si el rayo del LIDAR golpeó un obstáculo o no

            ray_dist = r if hit_obstacle else self.last_scan.range_max # Distancia medida si golpeó un obstáculo, sino distancia máxima del LIDAR
            n_steps = int(ray_dist / self.map_data.resolution) # Número de grillas a lo largo del rayo
            for step in range(n_steps):
                d = step * self.map_data.resolution
                fx = pose[0] + d * np.cos(angle)
                fy = pose[1] + d * np.sin(angle)
                fcol, frow = self.map_data.world_to_grid(fx, fy)
                if 0 <= frow < self.map_data.height and 0 <= fcol < self.map_data.width:
                    if self.dynamic_map[frow, fcol] > 0:  # Si la celda ya está marcada como ocupada, no se reduce su valor
                        self.dynamic_map[frow, fcol] = max(self.dynamic_map[frow, fcol] - 0.15, 0.0) # Reduce el valor de las celdas a lo largo del rayo, pero no la marca como libre si ya estaba ocupada

            if hit_obstacle: # Si el rayo del LIDAR golpeó un obstáculo, marca la celda correspondiente como ocupada en el mapa dinámico
                wx = pose[0] + r * np.cos(angle)
                wy = pose[1] + r * np.sin(angle)
                col, row = self.map_data.world_to_grid(wx, wy)

                if 0 <= row < self.map_data.height and 0 <= col < self.map_data.width:
                    if not self.inflated[row, col]: # Si la celda no está ocupada en el mapa inflado estático, suma 0.3 a la celda del mapa dinámico
                        self.dynamic_map[row, col] = min(self.dynamic_map[row, col] + 0.3, 1.0)

        curr = (self.dynamic_map > 0.5) # Marca las celdas ocupadas en el mapa dinámico después de la actualización
        if np.any(prev != curr): # Si hubo algún cambio en el mapa dinámico, se marca dynamic_changed como True
            self.dynamic_changed = True

    def obstacle_on_path(self):
        """
        Verifica si hay obstáculos en el camino planificado del robot.
        Devuelve True si se detecta un obstáculo en el camino, sino devuelve False.
        """
        if self.path is None:
            return False

        dynamic_occupied = self.dynamic_map > 0.5
        dynamic_inflated = path_planning.inflate_map(dynamic_occupied, self.radius_cells)

        pose = self.estimated_pose
        dists = [np.hypot(p[0] - pose[0], p[1] - pose[1]) for p in self.path] # Calcula la distancia desde la posición estimada del robot hasta cada punto en el camino planificado
        closest = np.argmin(dists) # Encuentra el índice del punto más cercano en el camino planificado al robot

        for point in self.path[closest:closest + 20]: # Verifica los próximos 20 puntos en el camino planificado para detectar obstáculos
            col, row = self.map_data.world_to_grid(point[0], point[1])
            if 0 <= row < self.map_data.height and 0 <= col < self.map_data.width:
                if dynamic_inflated[row, col]:
                    return True
        return False

# ------------------------------------------------------
# PUBLISHERS
# ------------------------------------------------------
    def publish_movement(self, linear_x: float, angular_z: float):
        """
        Publica un mensaje de velocidad lineal y angular para controlar el movimiento del robot.
        Parámetros:
        - linear_x: velocidad lineal en el eje x (m/s).
        - angular_z: velocidad angular alrededor del eje z (rad/s).
        """
        self.current_omega = angular_z
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.publisher_vel.publish(msg)

    def publish_pose(self, pose: list):
        """
        Publica un mensaje de pose estimada del robot.
        Parámetros:
        - pose: lista que contiene la posición (x, y) y orientación (theta) del robot.
        """
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])

        q = R.from_euler('z', pose[2]).as_quat()
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        self.publish_estimated_pose.publish(msg)

    def publish_planned_path(self):
        """
        Publica un mensaje de ruta planificada que contiene una lista de poses (x, y) que representan la trayectoria del robot hacia el objetivo.
        """
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        for (x, y) in self.path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            msg.poses.append(pose)

        self.publish_path.publish(msg)

    def publish_inflated(self, map_: np.ndarray):
        """
        Publica un mensaje de mapa inflado que representa las áreas ocupadas y libres del entorno, teniendo en cuenta los obstáculos estáticos y dinámicos.
        Las celdas ocupadas se representan con un valor de 100, las celdas libres con un valor de 0 y las celdas desconocidas con un valor de -1.
        Parámetros:
        - map_: matriz booleana que indica las celdas ocupadas.
        """
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.map_data.resolution
        msg.info.width = self.map_data.width
        msg.info.height = self.map_data.height
        msg.info.origin.position.x = self.map_data.origin_x
        msg.info.origin.position.y = self.map_data.origin_y

        grid = np.full(map_.shape, -1, dtype=np.int16)
        grid[self.map_data.free & ~map_] = 0
        grid[map_] = 100

        msg.data = grid.flatten().tolist()
        self.publish_inflated_map.publish(msg)

    def publish_dynamic(self):
        """
        Publica un mensaje que representa el mapa dinámico del entorno del robot.
        """
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.map_data.resolution
        msg.info.width = self.map_data.width
        msg.info.height = self.map_data.height
        msg.info.origin.position.x = self.map_data.origin_x
        msg.info.origin.position.y = self.map_data.origin_y

        grid = np.zeros(self.dynamic_map.shape, dtype=np.int8)
        grid[self.dynamic_map > 0.5] = 100
        grid[self.dynamic_map <= 0.5] = 0

        msg.data = grid.flatten().tolist()
        self.publish_dynamic_map_pub.publish(msg)

    def publish_particles(self):
        """
        Publica un mensaje que contiene la nube de partículas que representan la estimación de la posición del robot.
        Cada partícula se representa como una pose (x, y, theta) en el espacio.
        """
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        for p in self.particles:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])

            q = R.from_euler('z', float(p[2])).as_quat()
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            msg.poses.append(pose)

        self.publisher_particles.publish(msg)

# ------------------------------------------------------
# LOOP
# ------------------------------------------------------
    def loop(self):
        """
        Loop principal del nodo. Representa la máquina de estados que controla el comportamiento del robot.
        """
        if self.estimated_pose is not None:
            self.publish_pose(self.estimated_pose)
            self.publish_particles()

        if self.dynamic_changed: # Si el mapa dinámico ha cambiado, se recalcula el campo de probabilidad y se publica el mapa dinámico actualizado.
            combined_occupied = self.map_data.occupied | (self.dynamic_map > 0.5)
            self.likelihood_field = likelihood_field_B.precompute_likelihood_field(combined_occupied, self.sigma, self.map_data.resolution)
            self.publish_dynamic()
            self.dynamic_changed = False
    
        if self.state == IDLE:
            if self.inipos is not None: # Si el estado es IDLE, se verifica si la posición inicial está definida.
                if self.goal_dif: # Si la posición objetivo cambió, se cambia al estado de planificación
                    self.state = PLANNING
                else: # Si la posición objetivo no cambió, se cambia al estado de espera de objetivo
                    self.state = WAITING_GOAL
            return

        elif self.state == WAITING_GOAL: # Si el estado es WAITING_GOAL, se verifica si la posición objetivo cambió.
            if self.goal_dif: # Si cambió, se cambia al estado de planificación.
                self.goal_dif = False
                self.state = PLANNING
            return

        elif self.state == PLANNING: # Si el estado es PLANNING, se verifica si la posición objetivo o la posición inicial cambiaron.
            if self.goal_dif or self.inipos_dif: # Si cambiaron, se reinicia el estado de planificación.
                self.goal_dif = False
                self.inipos_dif = False

            self.plan_route()
            if self.path is not None: # Si se encontró un camino, se publica y se cambia al estado de navegación.
                self.publish_planned_path()
                self.replan_attempts = 0
                self.obstacle_cooldown = 10
                self.state = NAVIGATING
            else: # Si no se encontró un camino, se incrementa el contador de intentos de replanificación y se muestra un mensaje de advertencia. Si se superan los 30 intentos, se muestra un mensaje de error y se reinicia el estado a WAITING_GOAL.
                self.replan_attempts += 1
                self.get_logger().warn(f'No path found, intento {self.replan_attempts}')
                if self.replan_attempts > 30:
                    self.get_logger().error('No se puede llegar al objetivo')
                    self.replan_attempts = 0
                    self.state = WAITING_GOAL
            return

        elif self.state == NAVIGATING:
            if abs(self.current_omega) > 0.3: # Si el robot está girando rápidamente, se incrementa un contador para limitar la frecuencia de actualización del mapa dinámico.
                self.dynamic_update_counter += 1
                if self.dynamic_update_counter >= 10: # Si el contador alcanza 10, se actualiza el mapa dinámico y se reinicia el contador (cada 1 segundo).
                    self.dynamic_update_counter = 0
                    self.update_dynamic_map()
            else:
                self.dynamic_update_counter = 0
                self.update_dynamic_map()

            self.publish_dynamic()

            if self.goal_dif or self.inipos_dif: # Si la posición objetivo o la posición inicial han cambiado, se reinicia el estado de planificación.
                self.goal_dif = False
                self.inipos_dif = False
                self.state = PLANNING
                return

            if self.obstacle_cooldown > 0: # Si el contador de obstáculos está activo, se decrementa en cada iteración del bucle.
                self.obstacle_cooldown -= 1
            elif self.obstacle_on_path(): # Si se detecta un obstáculo en el camino, se reinicia el estado de planificación.
                self.publish_movement(0.0, 0.0)
                self.state = PLANNING
                return

            dist_to_goal = np.hypot(self.estimated_pose[0] - self.goal[0], self.estimated_pose[1] - self.goal[1])
            if dist_to_goal < 0.15: # Si el robot está cerca del objetivo, se detiene y cambia al estado de ajuste de ángulo.
                self.publish_movement(0.0, 0.0)
                self.state = ADJUSTING_ANGLE
                return

            v, omega, _ = path_following.pure_pursuit(self.estimated_pose, self.path, self.lookahead)
            self.publish_movement(v, omega)
            return

        elif self.state == ADJUSTING_ANGLE:
            if self.goal_dif or self.inipos_dif: # Si la posición objetivo o la posición inicial han cambiado, se reinicia el estado de planificación.
                self.goal_dif = False
                self.inipos_dif = False
                self.publish_movement(0.0, 0.0)
                self.state = PLANNING
                return

            raw_error = np.arctan2(np.sin(self.goal[2] - self.estimated_pose[2]), np.cos(self.goal[2] - self.estimated_pose[2]))
            error = raw_error - np.sign(raw_error) * self.rotation_error
            if abs(error) < 0.05: # Si el error de orientación es pequeño, se detiene y cambia al estado de espera de objetivo.
                self.publish_movement(0.0, 0.0)
                self.state = WAITING_GOAL
            else:
                omega = np.clip(1.5 * error, -0.8, 0.8)
                self.publish_movement(0.0, omega)
            return

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = nodo_b()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()