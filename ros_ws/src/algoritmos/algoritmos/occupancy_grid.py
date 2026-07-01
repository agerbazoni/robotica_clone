import numpy as np
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

def bresenham(r0, c0, r1, c1):
   rows = []
   cols = []
   dr = abs(r1 - r0)
   dc = abs(c1 - c0)
   sr = 1 if r0 < r1 else -1
   sc = 1 if c0 < c1 else -1
   err = dc - dr
   while True:
       rows.append(r0)
       cols.append(c0)
       if r0 == r1 and c0 == c1:
           break
       e2 = 2 * err
       if e2 > -dr:
           err -= dr
           c0 += sc
       if e2 < dc:
           err += dc
           r0 += sr
   return np.array(rows), np.array(cols)

class ParticleMap():
   def __init__(self, width:float, height:float, resolution:float, origin_x:float, origin_y:float):
       self.width = width
       self.height = height
       self.resolution = resolution
       self.origin_x = origin_x
       self.origin_y = origin_y

       # Grilla de log-odds, 0 = desconocido. log_odds = log(p / (1 - p))
       self.grid = np.zeros((height, width), dtype=np.int16)

       # Constantes para actualización
       self.l_occ = 3       # log-odds para celda ocupada.
       self.l_free = -1     # log-odds para celda libre
 
   def world_to_grid(self, x, y):
       """
       Retorna el punto (x, y) transformado a columna y fila de la grilla
       """
       col = int((x - self.origin_x) / self.resolution)
       row = int((y - self.origin_y) / self.resolution)
       return col, row

   def grid_to_world(self, col, row):
       """
       Dada una casilla, retorna las coordenadas del punto en la esquina inferior-izquierda.
       """
       # Para saber el punto a partir de una celda
       x = col * self.resolution + self.origin_x
       y = row * self.resolution + self.origin_y
       return x, y

   def _ensure_bounds(self, xs, ys, margin=10):
       """
       Mapa dinamico: expande la grilla (y corre el origen) para que todos los puntos
       del mundo (xs, ys) entren, dejando un margen de celdas. Solo reasigna memoria
       cuando algun punto se sale de los limites actuales.
       """
       min_col = int(np.floor((np.min(xs) - self.origin_x) / self.resolution))
       max_col = int(np.floor((np.max(xs) - self.origin_x) / self.resolution))
       min_row = int(np.floor((np.min(ys) - self.origin_y) / self.resolution))
       max_row = int(np.floor((np.max(ys) - self.origin_y) / self.resolution))

       pad_left   = max(0, margin - min_col)
       pad_right  = max(0, max_col + margin - (self.width - 1))
       pad_bottom = max(0, margin - min_row)
       pad_top    = max(0, max_row + margin - (self.height - 1))

       if not (pad_left or pad_right or pad_bottom or pad_top):
           return  # todo entra: no hace falta agrandar

       new_w = self.width + pad_left + pad_right
       new_h = self.height + pad_bottom + pad_top
       new_grid = np.zeros((new_h, new_w), dtype=self.grid.dtype)
       new_grid[pad_bottom:pad_bottom + self.height, pad_left:pad_left + self.width] = self.grid
       self.grid = new_grid
       self.width = new_w
       self.height = new_h
       # El origen es la esquina inferior-izquierda: se corre al agregar celdas en -x / -y
       self.origin_x -= pad_left * self.resolution
       self.origin_y -= pad_bottom * self.resolution

   def update(self, pose, msg, subsample = 1):
       # Recibe la pose de la partícula y un scan LIDAR, traza cada rayo desde la pose hasta el punto de impacto y actualiza las celdas
       x, y, theta = pose
       pose_col, pose_row = self.world_to_grid (x, y)

       angle_min, angle_increment, range_max, ranges, range_min, intensities = msg

       ranges = np.asarray(ranges)
       scan_angles = angle_min + np.arange(len(ranges)) * angle_increment

       # Submuestreo + filtrado de rayos inválidos, todo vectorizado
       r = ranges[::subsample]
       a = scan_angles[::subsample]
       valid = np.isfinite(r) & (r >= range_min)
       r, a = r[valid], a[valid]
       if r.size == 0:
           return

       # Endpoints y celdas de impacto para todos los rayos de una sola pasada
       endpoint_x = x + r * np.cos(theta + a)
       endpoint_y = y + r * np.sin(theta + a)

       # Mapa dinamico: agrandar la grilla para que entren la pose y todos los endpoints
       self._ensure_bounds(np.append(endpoint_x, x), np.append(endpoint_y, y))

       # El origen/tamano pueden haber cambiado: recalcular indices con el origen actual
       pose_col, pose_row = self.world_to_grid(x, y)
       endpoint_col = ((endpoint_x - self.origin_x) / self.resolution).astype(int)
       endpoint_row = ((endpoint_y - self.origin_y) / self.resolution).astype(int)

       hit = r < range_max  # rayos que efectivamente impactan (no alcanzan range_max)

       for i in range(r.size):
           row, col = bresenham(pose_row, pose_col, endpoint_row[i], endpoint_col[i])

           mask = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
           row, col = row[mask], col[mask]

           if len(row)==0: continue
           self.grid[row[:-1], col[:-1]] += self.l_free # celdas libres

           if hit[i]:
               self.grid[row[-1], col[-1]] += self.l_occ # celda ocupada
     
       np.clip(self.grid, -127, 127, out=self.grid) # PARA EVITAR OVERFLOW DE INT8

   def get_probability(self):
       # Convierte log odds a probabilidades
       grid_prob = 1 / (1 + np.exp(-self.grid.astype(float)))
       return grid_prob

   def to_occupancy_grid_msg(self, frame_id, stamp):
       # Empaqueta la grilla en un mensaje para publicar en ROS. Llama internamente a get_probability() y escala a [0, 100]
       msg = OccupancyGrid()
       msg.header = Header()
       msg.header.frame_id = frame_id
       msg.header.stamp = stamp

       msg.info.resolution = self.resolution
       msg.info.width = self.width
       msg.info.height = self.height
       msg.info.origin.position.x = self.origin_x
       msg.info.origin.position.y = self.origin_y
       msg.info.origin.position.z = 0.0

       prob = self.get_probability()

       result = np.full(self.grid.shape, -1, dtype=np.int8)
       known = np.abs(self.grid) > 0 # Mascara booleana para las celdas conocidas
       result[known] = (prob[known] * 100).astype(np.int8)

       msg.data = result.flatten().tolist()

       return msg

   def copy(self):
       # Crea un clon del mapa copiando solo el array numpy. Se usa en el resampling (cuando una partícula "buena" se duplica, su mapa también se tiene que duplicar).
       new = ParticleMap.__new__(ParticleMap)
       new.width = self.width
       new.height = self.height
       new.resolution = self.resolution
       new.origin_x = self.origin_x
       new.origin_y = self.origin_y
       new.l_occ = self.l_occ
       new.l_free = self.l_free
       new.grid = self.grid.copy()
       return new
