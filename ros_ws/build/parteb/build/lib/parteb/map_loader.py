import numpy as np
from PIL import Image
import yaml
import os

class MapData:
    """
    Carga un mapa desde .pgm+.yaml o desde .npz (save_map.py).
    """

    def __init__(self, path, occupied_thresh=0.65, free_thresh=0.25):
        ext = os.path.splitext(path)[1].lower()

        if ext in ('.yaml', '.yml'):
            self._load_yaml(path)
        elif ext == '.npz':
            self._load_npz(path, occupied_thresh, free_thresh)
        else:
            raise ValueError(f"Formato de mapa no soportado: {ext}")

        self.occupied = self.prob >= self.occupied_thresh
        self.free = self.prob <= self.free_thresh

    def _load_yaml(self, yaml_path):
        """
        Carga un mapa desde un archivo .yaml (y su .pgm).
        """
        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)

        self.resolution = meta['resolution']
        self.origin_x = meta['origin'][0]
        self.origin_y = meta['origin'][1]
        self.occupied_thresh = meta['occupied_thresh']
        self.free_thresh = meta['free_thresh']

        pgm_path = meta['image']
        if not os.path.isabs(pgm_path):
            pgm_path = os.path.join(os.path.dirname(yaml_path), pgm_path)

        img = np.array(Image.open(pgm_path), dtype=np.uint8)
        img = np.flipud(img)  # map_saver hace flipud al guardar, deshacerlo

        self.height, self.width = img.shape

        # Reconstruir probabilidad de ocupación [0.0, 1.0]
        # En .pgm: 254 = libre, 0 = ocupado, 205 = desconocido
        self.prob = np.full_like(img, 0.5, dtype=np.float64)
        known = img != 205
        self.prob[known] = 1.0 - img[known] / 255.0

    def _load_npz(self, path, occupied_thresh, free_thresh):
        """
        Carga un mapa desde un archivo .npz (que crea de la parte A save_map.py).
        """
        d = np.load(path)
        grid = d['grid']

        self.resolution = float(d['resolution'])
        self.origin_x = float(d['origin_x'])
        self.origin_y = float(d['origin_y'])
        self.occupied_thresh = occupied_thresh
        self.free_thresh = free_thresh
        self.height, self.width = grid.shape

        # -1 -> 0.5 (desconocido), 0 a 100 -> 0.0 a 1.0
        self.prob = np.where(grid < 0, 0.5, grid.astype(np.float64) / 100.0)

    def world_to_grid(self, x, y):
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return col, row

    def grid_to_world(self, col, row):
        x = col * self.resolution + self.origin_x
        y = row * self.resolution + self.origin_y
        return x, y