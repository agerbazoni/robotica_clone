#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
import math
import numpy as np
from scipy.ndimage import label

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from parteb.map_loader import MapData
from algoritmos import path_planning


def _centros_de_tramos(linea):
    """
    Dada una scanline booleana (celdas alcanzables o no), devuelve el índice del
    CENTRO de cada tramo contiguo alcanzable. Poner el waypoint en el centro del
    corredor —en vez de en un nodo de grilla fijo— garantiza cubrir corredores
    finos (que un nodo de grilla se saltearía) y deja el punto lo más lejos posible
    de ambas paredes (máximo margen = más seguro).
    """
    centros = []
    i, n = 0, len(linea)
    while i < n:
        if linea[i]:
            j = i
            while j < n and linea[j]:
                j += 1
            centros.append((i + j - 1) // 2)
            i = j
        else:
            i += 1
    return centros


def generar_waypoints_cobertura(map_data: MapData, spacing_m: float = 0.4, robot_footprint_m: float = 0.20):
    """
    Genera una lista de puntos (x, y) en coordenadas del mundo que cubren el espacio
    libre ALCANZABLE por el robot, ubicando los puntos en el CENTRO de cada corredor.

    Una celda es alcanzable con el mismo criterio que el planner de parteb: libre y
    fuera del mapa inflado con 'robot_footprint_m'. En vez de tomar los nodos de una
    grilla fija (que se saltea corredores finos si ningún nodo cae encima), se barren
    scanlines horizontales y verticales cada 'spacing_m' y en cada tramo alcanzable
    se coloca un waypoint en su punto medio. Así:
      - todo corredor cruzado por una scanline recibe un punto (incluye anillos y
        pasillos angostos por los que el robot pasa), y
      - los puntos quedan centrados = máximo margen a las paredes.

    Nota: 'robot_footprint_m' debe ser >= (radio_robot + margen) del planner
    (~0.165 m sim / ~0.23 m real); un poco más deja margen extra contra choques.

    El orden es nearest-neighbor (greedy): recorre un corredor entero antes de saltar
    a otro, reduciendo revisitas frente a un barrido serpentina.
    """
    step_cells = max(1, round(spacing_m / map_data.resolution))
    radius_cells = max(1, round(robot_footprint_m / map_data.resolution))

    # Mismo mapa de alcanzabilidad que usa el planner: libre y fuera del footprint.
    inflated = path_planning.inflate_map(map_data.occupied, radius_cells)
    reachable = map_data.free & ~inflated

    # Conservar solo la componente conexa de alcanzable más grande (el interior real
    # del laberinto) y descartar islas de "libre" espurias por ruido de mapeo que
    # quedan fuera de las paredes: si no, se generan waypoints inalcanzables que
    # mandarían al robot afuera del recinto. Conectividad de 8 vecinos para coincidir
    # con el A* del planner.
    estructura_8 = np.ones((3, 3), dtype=bool)
    etiquetas, n_componentes = label(reachable, structure=estructura_8)
    if n_componentes > 1:
        tamanos = np.bincount(etiquetas.ravel())
        tamanos[0] = 0  # el fondo (etiqueta 0) no cuenta
        reachable = etiquetas == tamanos.argmax()

    # Centro de cada corredor cruzado por las scanlines (usamos un set para dedup
    # entre las pasadas horizontal y vertical).
    celdas = set()
    for row in range(0, map_data.height, step_cells):
        for col in _centros_de_tramos(reachable[row, :]):
            celdas.add((row, col))
    for col in range(0, map_data.width, step_cells):
        for row in _centros_de_tramos(reachable[:, col]):
            celdas.add((row, col))

    if not celdas:
        return []

    # Orden determinista de arranque (esquina inferior) y luego nearest-neighbor.
    candidatos = [map_data.grid_to_world(col, row) for row, col in sorted(celdas)]
    restantes = candidatos[1:]
    ordenados = [candidatos[0]]
    while restantes:
        ux, uy = ordenados[-1]
        idx = min(range(len(restantes)),
                  key=lambda i: (restantes[i][0] - ux) ** 2 + (restantes[i][1] - uy) ** 2)
        ordenados.append(restantes.pop(idx))

    return ordenados


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
        self.declare_parameter("waypoint_spacing", 0.4)     # separación entre puntos de patrulla [m]
        self.declare_parameter("waypoint_footprint", 0.0)   # footprint del robot [m]; 0 -> auto según 'robot'
        self.declare_parameter("goal_reached_threshold", 0.20)  # radio para considerar "llegué" [m] (piso ~0.15 por nodo_b)

        map_path = self.get_parameter("map_path").get_parameter_value().string_value
        spacing = self.get_parameter("waypoint_spacing").get_parameter_value().double_value
        footprint = self.get_parameter("waypoint_footprint").get_parameter_value().double_value
        self.goal_reached_threshold = self.get_parameter("goal_reached_threshold").get_parameter_value().double_value

        # Debe ser >= (radio_robot + margen) del planner de parteb para que ningún
        # waypoint quede inalcanzable. Le damos un margen extra por encima del mínimo
        # del planner (0.165 sim / 0.23 real) para no colocar waypoints pegados a la
        # pared: con el error de localización, apuntar al borde de lo transitable hace
        # que el robot roce las paredes. 0.20/0.28 deja ~11 cm de margen físico.
        if footprint <= 0.0:
            footprint = 0.20 if robot == "simulado" else 0.28

        self.waypoints = []
        self.waypoint_idx = -1
        self.current_target = None

        # Última posición del cono (en frame 'map') que le mandamos al planificador
        # mientras estamos en NAVEGANDO_AL_CONO. Si llegamos a este punto y no
        # llegó una detección de visión más nueva, asumimos que perdimos el cono
        # de vista (oclusión, giro, falso positivo puntual) y volvemos a explorar.
        self.nav_cono_target = None

        # Bandera de "crédito": se prende cuando llega una detección nueva y se
        # consume (apaga) la primera vez que, estando cerca del target, la
        # revisamos. Evita el falso positivo de volver a EXPLORANDO por pura
        # carrera: llegar justo al target un instante antes de que la próxima
        # detección (que ya viene en camino) actualice nav_cono_target.
        self.deteccion_reciente_sin_revisar = False

        # True si, durante la persecución actual, alguna detección vino marcada
        # como Nivel 2 confirmado (no solo sospecha de Nivel 1). Se resetea cada
        # vez que arrancamos una persecución nueva. Antes de comprometernos a la
        # fase ciega de aproximación final exigimos que esto haya sido True al
        # menos una vez: si llegamos cerca solo con sospechas de Nivel 1 (nunca
        # confirmado), es más probable que sea otra cosa roja (ej. un cilindro
        # parcialmente tapado) y preferimos volver a explorar.
        self.cono_confirmado_en_esta_persecucion = False

        # --- Alineación final frente al cono (ESTADO_APROXIMACION_FINAL) ---
        # No usamos la orientación publicada por vision_node para esto (un cono es
        # simétrico, no tiene un "frente" real, y esa pose ni siquiera trae una
        # orientación confiable). En cambio comparamos, con el LIDAR, la distancia
        # mínima del lado izquierdo del cono frontal de rayos contra la del lado
        # derecho: si son parecidas, estamos centrados; si no, hay que girar.
        self.declare_parameter("alineacion_tolerancia_m", 0.03)  # diferencia izq/der tolerada [m]
        self.declare_parameter("alineacion_kp", 1.5)              # ganancia proporcional del giro
        self.declare_parameter("alineacion_velocidad_angular_max", 0.3)  # [rad/s] tope de seguridad
        self.alineacion_tolerancia_m = self.get_parameter("alineacion_tolerancia_m").get_parameter_value().double_value
        self.alineacion_kp = self.get_parameter("alineacion_kp").get_parameter_value().double_value
        self.alineacion_velocidad_angular_max = self.get_parameter("alineacion_velocidad_angular_max").get_parameter_value().double_value
        # Sub-estado de ESTADO_APROXIMACION_FINAL: al entrar, primero nos
        # alineamos de frente al cono (girando en el lugar, SIN avanzar) y recién
        # después empezamos a avanzar en línea recta hasta los 20cm.
        self.alineando_cono = False

        if map_path:
            try:
                map_data = MapData(map_path)
                self.waypoints = generar_waypoints_cobertura(map_data, spacing, footprint)
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
            # Logueamos una sola vez: al limpiar current_target, pose_callback deja
            # de re-disparar esta función (su guard corta cuando current_target is None).
            if self.current_target is not None:
                self.get_logger().warn('🏁 Recorrimos todos los waypoints de cobertura y no encontramos el cono.')
            self.current_target = None
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

    def reintentar_waypoint_actual(self):
        """
        Re-publica el waypoint de exploración en el que estábamos (sin avanzar
        el índice), para retomar la patrulla justo donde la habíamos dejado
        cuando salimos a perseguir al cono.
        """
        if self.current_target is None:
            # No había patrulla en curso (ej. ya la habíamos terminado antes de
            # ver el cono) o no hay waypoints: no hay nada que reintentar.
            self.enviar_siguiente_waypoint()
            return

        x, y = self.current_target
        goal_exploracion = PoseStamped()
        goal_exploracion.header.frame_id = 'map'
        goal_exploracion.header.stamp = self.get_clock().now().to_msg()
        goal_exploracion.pose.position.x = x
        goal_exploracion.pose.position.y = y
        goal_exploracion.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_exploracion)
        self.get_logger().info(
            f'Retomando waypoint {self.waypoint_idx + 1}/{len(self.waypoints)}: ({x:.2f}, {y:.2f})')

    def pose_callback(self, msg):
        if self.estado_actual == self.ESTADO_EXPLORANDO:
            if self.current_target is None:
                return

            dx = msg.pose.position.x - self.current_target[0]
            dy = msg.pose.position.y - self.current_target[1]
            distancia = math.hypot(dx, dy)

            if distancia <= self.goal_reached_threshold:
                self.get_logger().info('✅ Waypoint alcanzado. Sin cono a la vista, sigo patrullando.')
                self.enviar_siguiente_waypoint()
            return

        if self.estado_actual == self.ESTADO_NAVEGANDO_AL_CONO and self.nav_cono_target is not None:
            # OJO: acá NO usamos un timeout por segundos. Un mapa grande o con
            # muchos obstáculos puede tardar en llegar sin que eso signifique que
            # perdimos el cono. En cambio, comparamos contra la ÚLTIMA posición
            # del cono que conocemos.
            dx = msg.pose.position.x - self.nav_cono_target[0]
            dy = msg.pose.position.y - self.nav_cono_target[1]
            distancia = math.hypot(dx, dy)

            if distancia <= self.goal_reached_threshold:
                if self.deteccion_reciente_sin_revisar:
                    # Llegamos, pero hubo una detección nueva hace poco (puede
                    # estar en camino un target más cercano todavía). Le damos
                    # una vuelta más antes de decidir que lo perdimos.
                    self.deteccion_reciente_sin_revisar = False
                    return

                self.get_logger().info(
                    '🤔 Llegué a la última posición conocida del cono y no lo veo más. Vuelvo a EXPLORAR.')
                self.estado_actual = self.ESTADO_EXPLORANDO
                self.nav_cono_target = None
                self.reintentar_waypoint_actual()

    def vision_callback(self, msg):
        if self.estado_actual == self.ESTADO_EXPLORANDO:
            self.get_logger().info('¡👀 Cono a la vista! Cambiando a NAVEGANDO_AL_CONO.')
            self.estado_actual = self.ESTADO_NAVEGANDO_AL_CONO
            self.cono_confirmado_en_esta_persecucion = False

        if self.estado_actual == self.ESTADO_NAVEGANDO_AL_CONO:
            self.deteccion_reciente_sin_revisar = True

            # z=1.0 -> Nivel 2 confirmado (forma validada); z=0.0 -> solo
            # sospecha de Nivel 1 (ver nota en vision_node.py).
            confirmado_este_mensaje = msg.pose.position.z >= 0.5
            if confirmado_este_mensaje:
                self.cono_confirmado_en_esta_persecucion = True

            distancia_al_cono = msg.pose.position.x

            if distancia_al_cono < 0.60:
                if not self.cono_confirmado_en_esta_persecucion:
                    # Llegamos cerca pero nunca lo vimos confirmado (Nivel 2):
                    # veníamos solo por sospechas de Nivel 1. A esta distancia un
                    # cono real ya debería mostrar área y forma claras, así que
                    # asumimos falso positivo (ej. un cilindro parecido) y no nos
                    # comprometemos a la fase ciega de aproximación final.
                    self.get_logger().warn(
                        '🤨 Llegué cerca pero nunca confirmé que sea un cono (solo sospechas). '
                        'Puede ser otra cosa roja. Vuelvo a EXPLORAR.')
                    self.estado_actual = self.ESTADO_EXPLORANDO
                    self.nav_cono_target = None
                    self.reintentar_waypoint_actual()
                    return

                self.get_logger().info('⚠️ Cono a menos de 60cm. Apagando planificador, paso a LIDAR.')
                self.estado_actual = self.ESTADO_APROXIMACION_FINAL
                self.nav_cono_target = None
                # Al entrar a esta fase vamos "a ciegas" (sin visión, solo LIDAR):
                # primero nos alineamos de frente al cono sin avanzar, y recién
                # después empezamos a acercarnos en línea recta.
                self.alineando_cono = True
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
                    self.nav_cono_target = (goal_global.position.x, goal_global.position.y)
                    self.get_logger().info(f'🗺️ Ruta actualizada en el mapa: X={goal_global.position.x:.2f}, Y={goal_global.position.y:.2f}')
                    
                except Exception as e:
                    pass # Silenciamos el warning de tf2 para no ensuciar la terminal

    def scan_callback(self, msg):
        if self.estado_actual != self.ESTADO_APROXIMACION_FINAL:
            return

        rayos_10_grados = int(math.radians(10) / msg.angle_increment)
        # En el LaserScan, índice 0 = frente, ángulos crecientes hacia la izquierda
        # (CCW). Los primeros N rayos son "frente-hacia-la-derecha" y los últimos N
        # (justo antes de completar la vuelta) son "frente-hacia-la-izquierda".
        # Los separamos para poder comparar un lado contra el otro.
        rayos_derecha = msg.ranges[0:rayos_10_grados]
        rayos_izquierda = msg.ranges[-rayos_10_grados:]

        validos_derecha = [r for r in rayos_derecha if 0.0 < r < float('inf')]
        validos_izquierda = [r for r in rayos_izquierda if 0.0 < r < float('inf')]

        if self.alineando_cono:
            # Fase 1: girar en el lugar hasta quedar de frente al cono, SIN
            # avanzar todavía. Si falta un lado completo de lecturas válidas no
            # podemos estimar el error con confianza: nos quedamos quietos en
            # vez de girar a ciegas.
            if not (validos_izquierda and validos_derecha):
                self.cmd_vel_pub.publish(Twist())
                return

            diferencia = min(validos_izquierda) - min(validos_derecha)

            if abs(diferencia) <= self.alineacion_tolerancia_m:
                self.get_logger().info('✅ Alineado de frente al cono. Empiezo a avanzar.')
                self.alineando_cono = False
                self.cmd_vel_pub.publish(Twist())
                return

            vel_msg = Twist()
            # diferencia > 0 -> izquierda más lejos que derecha -> el cono está
            # corrido hacia la derecha -> giramos (angular.z negativo) hacia la derecha.
            angular = -self.alineacion_kp * diferencia
            angular = max(-self.alineacion_velocidad_angular_max,
                          min(self.alineacion_velocidad_angular_max, angular))
            vel_msg.angular.z = angular
            self.cmd_vel_pub.publish(vel_msg)
            return

        # Fase 2: ya alineados, avanzamos derecho hasta quedar a 20cm del cono.
        lecturas_validas = validos_derecha + validos_izquierda
        if not lecturas_validas:
            return

        distancia = min(lecturas_validas)
        vel_msg = Twist()

        if distancia <= 0.20:
            self.get_logger().info('🛑 ¡FRENANDO! Quedamos a 20cm exactos. Misión Cumplida.')
            self.cmd_vel_pub.publish(Twist())
            self.estado_actual = self.ESTADO_MISION_CUMPLIDA
        else:
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