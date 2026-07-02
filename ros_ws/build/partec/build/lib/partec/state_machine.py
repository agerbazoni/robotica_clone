#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
import math

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from parteb.map_loader import MapData


def generar_waypoints_cobertura(map_data: MapData, spacing_m: float = 1.0, clearance_m: float = 0.30):
    """
    Genera una lista de puntos (x, y) en coordenadas del mundo que cubren
    el espacio libre conocido del mapa, en barrido tipo "serpentina" (boustrophedon)
    para recorrer el laberinto entero de forma ordenada.

    Un punto se considera válido solo si él y una ventana de 'clearance_m' metros
    a su alrededor están completamente libres (evita mandar al robot a rozar paredes).
    """
    step_cells = max(1, round(spacing_m / map_data.resolution))
    clearance_cells = max(1, round(clearance_m / map_data.resolution))

    waypoints_por_fila = []
    filas = list(range(0, map_data.height, step_cells))

    for fila_idx, row in enumerate(filas):
        candidatos_fila = []
        for col in range(0, map_data.width, step_cells):
            r0, r1 = max(0, row - clearance_cells), min(map_data.height, row + clearance_cells + 1)
            c0, c1 = max(0, col - clearance_cells), min(map_data.width, col + clearance_cells + 1)
            ventana = map_data.free[r0:r1, c0:c1]

            # Toda la ventana debe ser espacio libre conocido (free ya excluye lo desconocido,
            # porque prob=0.5 no cumple free <= free_thresh salvo casos raros de threshold).
            if ventana.size > 0 and ventana.all():
                x, y = map_data.grid_to_world(col, row)
                candidatos_fila.append((x, y))

        # Barrido serpentina: alternamos el sentido en cada fila para minimizar
        # el recorrido total entre puntos consecutivos.
        if fila_idx % 2 == 1:
            candidatos_fila.reverse()

        waypoints_por_fila.extend(candidatos_fila)

    return waypoints_por_fila


class MisionConoStateMachine(Node):
    def __init__(self):
        super().__init__('state_machine_node')
        
        self.ESTADO_EXPLORANDO = 0
        self.ESTADO_NAVEGANDO_AL_CONO = 1
        self.ESTADO_APROXIMACION_FINAL = 2
        self.ESTADO_MISION_CUMPLIDA = 3
        
        self.estado_actual = self.ESTADO_EXPLORANDO
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter("robot", "simulado")
        robot = self.get_parameter("robot").get_parameter_value().string_value

        if robot not in ("real", "simulado"):
            self.get_logger().warn(f"Valor inválido para 'robot': '{robot}'. Usando 'simulado'.")
            robot = "simulado"

        # Parámetros de la exploración por cobertura
        self.declare_parameter("map_path", "")
        self.declare_parameter("waypoint_spacing", 1.0)     # separación entre puntos de patrulla [m]
        self.declare_parameter("waypoint_clearance", 0.30)  # distancia mínima a paredes [m]
        self.declare_parameter("goal_reached_threshold", 0.35)  # radio para considerar "llegué" [m]

        map_path = self.get_parameter("map_path").get_parameter_value().string_value
        spacing = self.get_parameter("waypoint_spacing").get_parameter_value().double_value
        clearance = self.get_parameter("waypoint_clearance").get_parameter_value().double_value
        self.goal_reached_threshold = self.get_parameter("goal_reached_threshold").get_parameter_value().double_value

        self.waypoints = []
        self.waypoint_idx = -1
        self.current_target = None

        if map_path:
            try:
                map_data = MapData(map_path)
                self.waypoints = generar_waypoints_cobertura(map_data, spacing, clearance)
                self.get_logger().info(f'Mapa cargado desde {map_path}. {len(self.waypoints)} waypoints de cobertura generados.')
            except Exception as e:
                self.get_logger().error(f'No se pudo cargar el mapa ({map_path}): {e}. Exploración deshabilitada.')
        else:
            self.get_logger().warn('No se especificó "map_path": no se generarán waypoints de cobertura.')

        # Tópicos internos (no dependen del robot -> sin prefijo tb4_0)
        self.vision_sub = self.create_subscription(
            PoseStamped, '/vision/cono_pose_relativa', self.vision_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Tópicos del robot (driver real vs. simulado -> cambian de nombre)
        if robot == "simulado":
            self.scan_sub = self.create_subscription(
                LaserScan, '/scan', self.scan_callback, 10)
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.pose_sub = self.create_subscription(
                PoseStamped, '/estimated_pose', self.pose_callback, 10)
        else:
            # Los sensores del TB publican BEST_EFFORT: el subscriber debe matchear o no recibe nada
            sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.scan_sub = self.create_subscription(
                LaserScan, '/tb4_0/scan', self.scan_callback, sensor_qos)
            self.cmd_vel_pub = self.create_publisher(Twist, '/tb4_0/cmd_vel', 10)
            self.pose_sub = self.create_subscription(
                PoseStamped, '/tb4_0/estimated_pose', self.pose_callback, 10)

        self.get_logger().info(f'Cerebro iniciado (robot={robot}). Estado: EXPLORANDO laberinto...')
        
        # Le damos 3 segundos al nodo para que levante y mandamos el primer destino
        self.timer = self.create_timer(3.0, self.iniciar_exploracion)

    def iniciar_exploracion(self):
        # Corre una sola vez, para arrancar el recorrido de patrulla
        self.timer.destroy()
        if self.estado_actual == self.ESTADO_EXPLORANDO:
            self.enviar_siguiente_waypoint()

    def enviar_siguiente_waypoint(self):
        self.waypoint_idx += 1

        if self.waypoint_idx >= len(self.waypoints):
            self.get_logger().warn('🏁 Recorrimos todos los waypoints de cobertura y no encontramos el cono.')
            return

        x, y = self.waypoints[self.waypoint_idx]
        self.current_target = (x, y)

        goal_exploracion = PoseStamped()
        goal_exploracion.header.frame_id = 'map'
        goal_exploracion.header.stamp = self.get_clock().now().to_msg()
        goal_exploracion.pose.position.x = x
        goal_exploracion.pose.position.y = y
        goal_exploracion.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_exploracion)
        self.get_logger().info(
            f'Waypoint {self.waypoint_idx + 1}/{len(self.waypoints)} enviado: ({x:.2f}, {y:.2f})')

    def pose_callback(self, msg):
        if self.estado_actual != self.ESTADO_EXPLORANDO or self.current_target is None:
            return

        dx = msg.pose.position.x - self.current_target[0]
        dy = msg.pose.position.y - self.current_target[1]
        distancia = math.hypot(dx, dy)

        if distancia <= self.goal_reached_threshold:
            self.get_logger().info('✅ Waypoint alcanzado. Sin cono a la vista, sigo patrullando.')
            self.enviar_siguiente_waypoint()

    def vision_callback(self, msg):
        if self.estado_actual == self.ESTADO_EXPLORANDO:
            self.get_logger().info('¡👀 Cono a la vista! Cambiando a NAVEGANDO_AL_CONO.')
            self.estado_actual = self.ESTADO_NAVEGANDO_AL_CONO
            
        if self.estado_actual == self.ESTADO_NAVEGANDO_AL_CONO:
            distancia_al_cono = msg.pose.position.x 
            
            if distancia_al_cono < 0.60:
                self.get_logger().info('⚠️ Cono a menos de 60cm. Apagando planificador, paso a LIDAR.')
                self.estado_actual = self.ESTADO_APROXIMACION_FINAL
            else:
                try:
                    transformacion = self.tf_buffer.lookup_transform(
                        'map', msg.header.frame_id, rclpy.time.Time())
                    
                    goal_global = tf2_geometry_msgs.do_transform_pose(msg.pose, transformacion)
                    
                    goal_msg_map = PoseStamped()
                    goal_msg_map.header.frame_id = 'map'
                    goal_msg_map.header.stamp = self.get_clock().now().to_msg()
                    goal_msg_map.pose = goal_global
                    
                    self.goal_pub.publish(goal_msg_map)
                    self.get_logger().info(f'🗺️ Ruta actualizada en el mapa: X={goal_global.position.x:.2f}, Y={goal_global.position.y:.2f}')
                    
                except Exception as e:
                    pass # Silenciamos el warning de tf2 para no ensuciar la terminal

    def scan_callback(self, msg):
        if self.estado_actual == self.ESTADO_APROXIMACION_FINAL:
            rayos_10_grados = int(math.radians(10) / msg.angle_increment)
            frente = msg.ranges[0:rayos_10_grados] + msg.ranges[-rayos_10_grados:]
            lecturas_validas = [r for r in frente if 0.0 < r < float('inf')]
            
            if lecturas_validas:
                distancia = min(lecturas_validas)
                if distancia <= 0.20:
                    self.get_logger().info('🛑 ¡FRENANDO! Quedamos a 20cm exactos. Misión Cumplida.')
                    vel_msg = Twist()
                    self.cmd_vel_pub.publish(vel_msg)
                    self.estado_actual = self.ESTADO_MISION_CUMPLIDA
                else:
                    vel_msg = Twist()
                    vel_msg.linear.x = 0.1 
                    self.cmd_vel_pub.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    sm_node = MisionConoStateMachine()
    try:
        rclpy.spin(sm_node)
    except KeyboardInterrupt:
        pass
    sm_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()