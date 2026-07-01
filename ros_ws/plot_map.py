#!/usr/bin/env python3
"""
Visualiza un mapa de ocupacion guardado (CSV o .npy) con matplotlib.

Uso:
    python3 plot_map.py                          # carga mapa.csv
    python3 plot_map.py mapa.npy
    python3 plot_map.py mapa.csv --resolution 0.02 --origin -2 -2   # ejes en metros

Convencion de valores (OccupancyGrid de ROS):
    -1   = desconocido  -> gris
     0   = libre        -> blanco
   100   = ocupado      -> negro
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_grid(path):
    """Devuelve (grid, resolution, origin) — resolution/origin son None si el formato no los trae."""
    if path.endswith(".npz"):
        d = np.load(path)
        return d["grid"], float(d["resolution"]), (float(d["origin_x"]), float(d["origin_y"]))
    if path.endswith(".npy"):
        return np.load(path), None, None
    return np.loadtxt(path, delimiter=",", dtype=np.int16), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="mapa.csv", help="archivo .csv o .npy")
    ap.add_argument("--resolution", type=float, default=None, help="metros por celda (para ejes en m)")
    ap.add_argument("--origin", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="origen (x, y) en metros de la esquina inferior-izquierda")
    ap.add_argument("--save", default=None, help="guardar la figura a un archivo (png) en vez de mostrarla")
    args = ap.parse_args()

    grid, res, origin = load_grid(args.path)
    grid = grid.astype(float)

    # Si el archivo (.npz) trae metadata, usarla salvo que la pisen por linea de comandos
    if args.resolution is None:
        args.resolution = res
    if args.origin is None:
        args.origin = origin

    # A escala de grises: desconocido -> 0.5, libre(0) -> 1 (blanco), ocupado(100) -> 0 (negro)
    img = np.where(grid < 0, 0.5, 1.0 - grid / 100.0)

    # data va de la esquina inferior-izquierda hacia arriba: flipud para que "arriba" sea +y
    img = np.flipud(img)

    # extent en metros si se dan resolution y origin; si no, en celdas
    extent = None
    xlabel, ylabel = "columna [celda]", "fila [celda]"
    if args.resolution is not None and args.origin is not None:
        h, w = grid.shape
        ox, oy = args.origin
        extent = [ox, ox + w * args.resolution, oy, oy + h * args.resolution]
        xlabel, ylabel = "x [m]", "y [m]"

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, extent=extent, origin="upper")

    # Marcar el origen del mundo (0,0) = donde arranco el robot. Necesita ejes en metros.
    if extent is not None:
        plt.plot(0.0, 0.0, marker="o", color="red", markersize=8, label="origen (0,0)")
        plt.legend(loc="upper right")
    else:
        print("Nota: sin resolution/origin no se pueden poner los ejes en metros ni marcar (0,0). "
              "Carga el .npz (o pasa --resolution y --origin).")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Mapa de ocupacion ({grid.shape[0]}x{grid.shape[1]} celdas)")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Figura guardada en {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
