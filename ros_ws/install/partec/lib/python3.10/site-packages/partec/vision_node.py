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

        # Filtro geométrico por altura real estimada: como conocemos la altura del
        # cono y la altura de la cámara, podemos invertir la proyección de la cámara
        # para estimar cuánto mide realmente lo que detectamos. Si "algo rojo" mide
        # 1.60m de alto (una persona), no es un cono aunque el HSV matchee.
        self.declare_parameter("cone_height", 0.30)            # [m] altura real del cono
        self.declare_parameter("cone_height_tolerance", 0.15)  # [m] margen de error tolerado
        self.declare_parameter("max_cone_distance", 4.0)       # [m] descarta detecciones más lejanas que esto
        self.cone_height = self.get_parameter("cone_height").get_parameter_value().double_value
        self.cone_height_tolerance = self.get_parameter("cone_height_tolerance").get_parameter_value().double_value
        self.max_cone_distance = self.get_parameter("max_cone_distance").get_parameter_value().double_value
        
        # Variables de estado
        self.br = CvBridge()
        self.intrinsics = None
        self.camera_height = 0.30
        
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

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # --- NIVEL 1: EL AMAGUE (Sospechoso) ---
            if 20 < area <= 100:
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 255), 2) # Amarillo
                cv2.putText(current_frame, "Sospechoso", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                # Todavía no calculamos distancia ni publicamos, solo marcamos interés.

            # --- NIVEL 2: CONFIRMADO ---
            elif area > 100:
                aspect_ratio = float(w) / h
                if 0.10 < aspect_ratio < 0.40:
                    cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Verde
                    cv2.putText(current_frame, "Cono Confirmado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if self.intrinsics is not None:
                        u, v = x + (w / 2.0), y + h
                        
                        if v >= image_height - 5:
                            print("⚠️ Cono muy cerca: Ciego para estimar distancia.")
                        elif (v - self.intrinsics['cy']) != 0:
                            Z = (self.camera_height * self.intrinsics['fy']) / (v - self.intrinsics['cy'])
                            X = ((u - self.intrinsics['cx']) * Z) / self.intrinsics['fx']
                            
                            if 0.1 < Z < self.max_cone_distance:
                                # Filtro geométrico por altura: invertimos la misma proyección
                                # que usamos para Z, pero con la fila SUPERIOR del bounding box (y),
                                # para estimar cuánto mide realmente el objeto detectado.
                                altura_estimada = self.camera_height - ((y - self.intrinsics['cy']) * Z / self.intrinsics['fy'])
                                altura_min = self.cone_height - self.cone_height_tolerance
                                altura_max = self.cone_height + self.cone_height_tolerance

                                if not (altura_min < altura_estimada < altura_max):
                                    # Es rojo, tiene forma de cono en 2D, pero mide otra cosa
                                    # en el mundo real (ej. una persona) -> lo descartamos.
                                    continue

                                # LÓGICA DE PUBLICACIÓN CON UMBRAL
                                publicar = False
                                if self.last_published_x is None:
                                    publicar = True
                                else:
                                    # Calculamos cuánto varió la medición desde la última publicación
                                    distancia_movida = math.hypot(X - self.last_published_x, Z - self.last_published_z)
                                    if distancia_movida > 0.20: # Si varió más de 20cm, actualizamos
                                        publicar = True
                                
                                if publicar:
                                    goal_msg = PoseStamped()
                                    goal_msg.header.stamp = self.get_clock().now().to_msg()
                                    
                                    # Transformación básica al marco del robot (X adelante, Y izquierda)
                                    goal_msg.header.frame_id = "camera_link" 
                                    goal_msg.pose.position.x = Z
                                    goal_msg.pose.position.y = -X
                                    goal_msg.pose.position.z = 0.0
                                    
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