#!/usr/bin/env python3
"""
Guarda el ultimo mapa publicado (OccupancyGrid) en CSV (y .npy).

Uso:
    python3 save_map.py                      # tópico por defecto /tb4_0/map -> mapa.csv
    python3 save_map.py /map mi_mapa         # tópico y nombre de salida custom

El tópico del mapa esta latcheado (TRANSIENT_LOCAL), asi que alcanza con correr
esto MIENTRAS el nodo slam sigue vivo (toma el ultimo mapa publicado al instante).
"""
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid


class MapSaver(Node):
    def __init__(self, topic, out):
        super().__init__('map_saver')
        self.out = out
        # Mismo QoS que publica el slam: latcheado y confiable
        qos = QoSProfile(depth=1,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(OccupancyGrid, topic, self.cb, qos)
        self.get_logger().info(f"Esperando un mensaje en {topic}...")

    def cb(self, msg):
        w, h = msg.info.width, msg.info.height
        # data es row-major desde la esquina inferior-izquierda (fila 0 = abajo)
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)

        res = msg.info.resolution
        o = msg.info.origin.position

        # .npz: grilla + metadata. SIN el origen, quien cargue el mapa no sabe donde
        # esta apoyada la grilla (con el mapa dinamico NO esta centrada en el robot),
        # y asumir "centrado" desfasa todo el mapa respecto a la odometria real.
        np.savez(f"{self.out}.npz",
                 grid=grid,
                 resolution=res,
                 origin_x=o.x,
                 origin_y=o.y)

        # CSV solo para inspeccion visual (no lleva metadata)
        np.savetxt(f"{self.out}.csv", grid, fmt="%d", delimiter=",")

        self.get_logger().info(
            f"Guardado {self.out}.npz (+ .csv)  |  {h}x{w} celdas, "
            f"resolucion={res} m, origin=({o.x:.3f}, {o.y:.3f})")
        rclpy.shutdown()


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/tb4_0/map"
    out = sys.argv[2] if len(sys.argv) > 2 else "mapa"
    rclpy.init()
    rclpy.spin(MapSaver(topic, out))


if __name__ == "__main__":
    main()
