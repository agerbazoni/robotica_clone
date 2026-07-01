#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
import math

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

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
        
        self.vision_sub = self.create_subscription(
            PoseStamped, '/vision/cono_pose_relativa', self.vision_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/tb4_0/scan', self.scan_callback, 10)
            
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info('Cerebro iniciado. Estado: EXPLORANDO laberinto...')
        
        # Le damos 3 segundos al nodo para que levante y mandamos el primer destino
        self.timer = self.create_timer(3.0, self.iniciar_exploracion)

    def iniciar_exploracion(self):
        # Esta función corre una sola vez para que el robot empiece a moverse
        if self.estado_actual == self.ESTADO_EXPLORANDO:
            goal_exploracion = PoseStamped()
            goal_exploracion.header.frame_id = 'map'
            goal_exploracion.header.stamp = self.get_clock().now().to_msg()
            
            # Coordenada arbitraria en tu mapa para patrullar (ajustala según tu laberinto)
            goal_exploracion.pose.position.x = 2.0
            goal_exploracion.pose.position.y = 1.0
            goal_exploracion.pose.orientation.w = 1.0
            
            self.goal_pub.publish(goal_exploracion)
            self.get_logger().info('Waypoint de exploración enviado a Nav2.')
            
            # Destruimos el timer para que no siga mandando este punto
            self.timer.destroy()

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