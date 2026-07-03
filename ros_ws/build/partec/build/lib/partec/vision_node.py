#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped # NUEVO IMPORT PARA EL GOAL
from cv_bridge import CvBridge
import cv2
import numpy as np
import math # NUEVO IMPORT PARA CALCULAR DISTANCIAS

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter("robot", "simulado")
        robot = self.get_parameter("robot").get_parameter_value().string_value

        if robot not in ("real", "simulado"):
            self.get_logger().warn(f"Valor inválido para 'robot': '{robot}'. Usando 'simulado'.")
            robot = "simulado"

        # Tópicos de cámara configurables por launch (útil en simulado, donde
        # el nombre depende del mundo/URDF que se esté usando).
        self.declare_parameter("camera_topic_sim", "/camera/image_raw")
        self.declare_parameter("camera_info_topic_sim", "/camera/camera_info")

        if robot == "simulado":
            image_topic = self.get_parameter("camera_topic_sim").get_parameter_value().string_value
            camera_info_topic = self.get_parameter("camera_info_topic_sim").get_parameter_value().string_value

            self.subscription = self.create_subscription(
                Image, image_topic, self.image_callback, 10)
            self.camera_info_sub = self.create_subscription(
                CameraInfo, camera_info_topic, self.camera_info_callback, 1)
        else:
            # Los sensores del TB publican BEST_EFFORT: el subscriber debe matchear o no recibe nada
            sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

            self.subscription = self.create_subscription(
                Image, '/tb4_0/oakd/rgb/preview/image_raw', self.image_callback, sensor_qos)
            self.camera_info_sub = self.create_subscription(
                CameraInfo, '/tb4_0/oakd/rgb/preview/camera_info', self.camera_info_callback, sensor_qos)

        # PUBLICADOR DEL GOAL (interno, no depende del robot -> sin prefijo tb4_0)
        self.goal_pub = self.create_publisher(PoseStamped, '/vision/cono_pose_relativa', 10)

        # Frame en el que se publica la pose del cono. El cerebro hace
        # lookup_transform('map', <este frame>), asi que debe existir en el arbol de TF:
        #   - simulado: 'camera_link'
        #   - real: el frame optico del oakd (namespaceado, p.ej. tb4_0/oakd_rgb_camera_optical_frame),
        #     que varia segun URDF/namespace -> por defecto lo tomamos del header del
        #     camera_info (robusto). Se puede forzar con el parametro 'camera_frame'.
        self.declare_parameter("camera_frame", "")
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        if not self.camera_frame and robot == "simulado":
            self.camera_frame = "camera_link"
        # Si quedo fijo (sim o override manual) no se pisa con el frame del camera_info.
        self.camera_frame_fijo = bool(self.camera_frame)

        # Filtro geométrico por altura real estimada: como conocemos la altura del
        # cono y la altura de la cámara, podemos invertir la proyección de la cámara
        # para estimar cuánto mide realmente lo que detectamos. Si "algo rojo" mide
        # 1.60m de alto (una persona), no es un cono aunque el HSV matchee.
        self.declare_parameter("cone_height", 0.35)             # [m] altura real del cono (medido: ~35cm)
        self.declare_parameter("cone_height_tolerance", 0.15)   # [m] margen de error tolerado (filtro nivel 2)
        self.declare_parameter("max_cone_distance", 4.0)        # [m] descarta detecciones más lejanas que esto
        self.cone_height = self.get_parameter("cone_height").get_parameter_value().double_value
        self.cone_height_tolerance = self.get_parameter("cone_height_tolerance").get_parameter_value().double_value
        self.max_cone_distance = self.get_parameter("max_cone_distance").get_parameter_value().double_value

        # Altura de la cámara sobre el piso: varía según el robot (chasis distinto),
        # por eso queda configurable por parámetro en vez de hardcodeada. Si no se
        # pasa por launch (<=0.0), usamos un default razonable según 'robot'.
        self.declare_parameter("camera_height", 0.0)
        camera_height_param = self.get_parameter("camera_height").get_parameter_value().double_value
        if camera_height_param <= 0.0:
            camera_height_param = 0.084 if robot == "simulado" else 0.23
        self.camera_height = camera_height_param

        # --- Confirmación multi-frame ---
        # Un solo frame que pase los filtros puede ser ruido (reflejo, sombra, etc).
        # Exigimos varios frames CONSECUTIVOS válidos antes de confiar en la detección
        # y recién ahí habilitar la publicación. N chico reacciona rápido pero es más
        # sensible a ruido puntual; N grande es más robusto pero tarda más en reaccionar
        # y puede perder un cono real que aparece y desaparece por oclusiones cortas.
        self.declare_parameter("confirmaciones_necesarias", 3)
        self.confirmaciones_necesarias = self.get_parameter("confirmaciones_necesarias").get_parameter_value().integer_value
        self.frames_confirmados_consecutivos = 0

        # --- Override de Nivel 1 (objeto parcialmente oculto) ---
        # Un cono tapado a medias puede no llegar al área/aspect-ratio de Nivel 2,
        # pero si aun así podemos estimar su altura real y cae en una banda MUY
        # angosta (más estricta que la de Nivel 2), es muy probable que sea el cono
        # real y no otra cosa roja. Pedimos además un área mínima más alta que el
        # piso genérico de "Sospechoso" (20px), porque con muy pocos píxeles el
        # bounding box es ruidoso y la altura estimada no es confiable.
        self.declare_parameter("nivel1_area_minima", 50)
        self.declare_parameter("cone_height_min_estricta", 0.30)
        self.declare_parameter("cone_height_max_estricta", 0.35)
        self.nivel1_area_minima = self.get_parameter("nivel1_area_minima").get_parameter_value().integer_value
        self.cone_height_min_estricta = self.get_parameter("cone_height_min_estricta").get_parameter_value().double_value
        self.cone_height_max_estricta = self.get_parameter("cone_height_max_estricta").get_parameter_value().double_value

        # --- Diferenciar cono de cilindro ---
        # Un cono es angosto arriba y ancho en la base; un cilindro tiene el mismo
        # ancho en toda su altura. Comparamos el ancho del blob en su franja superior
        # contra la inferior; si la base no es sensiblemente más ancha que la punta,
        # no lo tratamos como cono aunque el aspect ratio del bounding box matchee.
        self.declare_parameter("min_relacion_ancho_base_punta", 1.4)
        self.min_relacion_ancho_base_punta = self.get_parameter("min_relacion_ancho_base_punta").get_parameter_value().double_value

        # Variables de estado
        self.br = CvBridge()
        self.intrinsics = None

        # Memoria para no spamear el Goal
        self.last_published_x = None
        self.last_published_z = None
        
        self.get_logger().info('Nodo de visión iniciado. Esperando datos...')

    def camera_info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {
                'fx': msg.k[0], 'cx': msg.k[2], 'fy': msg.k[4], 'cy': msg.k[5]
            }
            self.get_logger().info('¡Matriz intrínseca calibrada!')

        # En el real, adoptar el frame real de la camara del header (a menos que se haya
        # forzado por parametro). Asi el cerebro puede transformar la pose del cono.
        if not self.camera_frame_fijo and msg.header.frame_id:
            self.camera_frame = msg.header.frame_id
            self.camera_frame_fijo = True
            self.get_logger().info(f"Frame de cámara detectado: '{self.camera_frame}'")

    def _es_forma_de_cono(self, red_mask, x, y, w, h):
        """
        Compara el ancho del blob en su franja superior (20% de arriba) contra
        su franja inferior (20% de abajo), dentro de la máscara binaria ya
        umbralizada. Un cono real es angosto en la punta y ancho en la base;
        un cilindro (o un caño, una lata acostada rojos, etc.) mantiene un
        ancho similar en toda su altura. Devuelve True si el perfil es
        compatible con un cono.
        """
        banda = max(1, int(h * 0.2))
        roi = red_mask[y:y + h, x:x + w]
        if roi.shape[0] < 2:
            return True  # blob demasiado chato como para evaluar perfil, no descartamos por esto

        franja_superior = roi[0:banda, :]
        franja_inferior = roi[max(0, h - banda):h, :]

        ancho_superior = int(np.count_nonzero(franja_superior.any(axis=0)))
        ancho_inferior = int(np.count_nonzero(franja_inferior.any(axis=0)))

        if ancho_inferior == 0:
            return False  # no hay base detectada, algo raro con el contorno

        if ancho_superior == 0:
            return True  # punta cerrada del todo: caso típico de un cono bien centrado

        relacion = ancho_inferior / ancho_superior
        return relacion >= self.min_relacion_ancho_base_punta

    def image_callback(self, data):
        try:
            current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except Exception as e:
            return

        image_height = current_frame.shape[0]
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
        mask1, mask2 = cv2.inRange(hsv_frame, lower_red1, upper_red1), cv2.inRange(hsv_frame, lower_red2, upper_red2)
        red_mask = cv2.add(mask1, mask2)

        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Se vuelve True si en ESTE frame encontramos una detección que pasó
        # todos los filtros (forma/área/altura). Alimenta el contador de
        # confirmación multi-frame.
        deteccion_valida = False
        X = Z = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            candidato = None  # 'confirmado' (nivel 2) o 'nivel1_override'

            # --- NIVEL 1: EL AMAGUE (Sospechoso) ---
            if 20 < area <= 100:
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Amarillo
                cv2.putText(current_frame, "Sospechoso", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if area > self.nivel1_area_minima:
                    # Objeto chico o parcialmente tapado: no confiamos en su forma 2D,
                    # pero si la altura real estimada cae en una banda MUY angosta
                    # (30-35cm), es muy probable que sea el cono y no otra cosa roja.
                    candidato = 'nivel1_override'
                # Si el área es <= nivel1_area_minima, es demasiado ruidoso como para
                # confiar en ninguna estimación (ni de altura): lo dejamos como
                # "Sospechoso" nomás y seguimos con el próximo contorno.

            # --- NIVEL 2: CONFIRMADO ---
            elif area > 100:
                aspect_ratio = float(w) / h
                if 0.10 < aspect_ratio < 0.40:
                    if not self._es_forma_de_cono(red_mask, x, y, w, h):
                        # Rojo, alto y angosto, pero con ancho parejo en toda su
                        # altura -> más compatible con un cilindro que con un cono.
                        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 140, 255), 2)  # Naranja
                        cv2.putText(current_frame, "Forma no compatible", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                        continue

                    cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Verde
                    cv2.putText(current_frame, "Cono Confirmado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    candidato = 'confirmado'

            if candidato is None or self.intrinsics is None:
                continue

            u, v = x + (w / 2.0), y + h

            if v >= image_height - 5:
                print("⚠️ Cono muy cerca: Ciego para estimar distancia.")
                continue
            if (v - self.intrinsics['cy']) == 0:
                continue

            Z_cand = (self.camera_height * self.intrinsics['fy']) / (v - self.intrinsics['cy'])
            X_cand = ((u - self.intrinsics['cx']) * Z_cand) / self.intrinsics['fx']

            if not (0.1 < Z_cand < self.max_cone_distance):
                continue

            # Filtro geométrico por altura: invertimos la misma proyección
            # que usamos para Z, pero con la fila SUPERIOR del bounding box (y),
            # para estimar cuánto mide realmente el objeto detectado.
            altura_estimada = self.camera_height - ((y - self.intrinsics['cy']) * Z_cand / self.intrinsics['fy'])

            if candidato == 'nivel1_override':
                if not (self.cone_height_min_estricta < altura_estimada < self.cone_height_max_estricta):
                    continue
            else:
                altura_min = self.cone_height - self.cone_height_tolerance
                altura_max = self.cone_height + self.cone_height_tolerance
                if not (altura_min < altura_estimada < altura_max):
                    # Es rojo, tiene forma de cono en 2D, pero mide otra cosa
                    # en el mundo real (ej. una persona) -> lo descartamos.
                    continue

            # Este contorno pasó todos los filtros: nos quedamos con él y
            # cortamos el loop (no elegimos "el mejor" entre varios candidatos
            # del mismo frame, nos alcanza con el primero que confirma).
            X, Z = X_cand, Z_cand
            deteccion_valida = True
            break

        if deteccion_valida:
            self.frames_confirmados_consecutivos += 1
        else:
            self.frames_confirmados_consecutivos = 0

        if deteccion_valida and self.frames_confirmados_consecutivos >= self.confirmaciones_necesarias:
            # LÓGICA DE PUBLICACIÓN CON UMBRAL
            publicar = False
            if self.last_published_x is None:
                publicar = True
            else:
                # Calculamos cuánto varió la medición desde la última publicación
                distancia_movida = math.hypot(X - self.last_published_x, Z - self.last_published_z)
                if distancia_movida > 0.20:  # Si varió más de 20cm, actualizamos
                    publicar = True

            if publicar:
                goal_msg = PoseStamped()
                goal_msg.header.stamp = self.get_clock().now().to_msg()

                # Transformación básica al marco del robot (X adelante, Y izquierda)
                goal_msg.header.frame_id = self.camera_frame
                goal_msg.pose.position.x = Z
                goal_msg.pose.position.y = -X
                # Reusamos 'z' (sin uso real en esta pose 2D) como flag de
                # confianza: 1.0 = confirmado por Nivel 2 (forma+aspect_ratio
                # validados), 0.0 = solo sospecha de Nivel 1 (override por altura,
                # sin validar forma porque el objeto está parcial/ocluido). Así
                # state_machine.py puede decidir si confía lo suficiente antes de
                # comprometerse a la fase ciega de aproximación final.
                goal_msg.pose.position.z = 1.0 if candidato == 'confirmado' else 0.0
                # Orientación identidad válida: el ángulo real de aproximación NO se
                # decide acá (no tenemos forma de saber hacia dónde "mira" un cono
                # simétrico); la alineación final la resuelve el LIDAR en
                # ESTADO_APROXIMACION_FINAL, en state_machine.py.
                goal_msg.pose.orientation.w = 1.0

                self.goal_pub.publish(goal_msg)

                self.last_published_x = X
                self.last_published_z = Z
                print(f"🚀 [Goal Publicado] Adelante={Z:.2f}m, Lateral={X:.2f}m")

        display_frame = cv2.resize(current_frame, (640, 480))
        cv2.imshow("Camara TurtleBot4", display_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionNode()
    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    vision_node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()