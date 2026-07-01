import numpy as np
from scipy.spatial import cKDTree

LOG_MIN = np.log(1e-6)  # piso de log-probabilidad para endpoints muy lejos de cualquier obstaculo


class LikelihoodField:
    """
    Likelihood field implementado con un KD-tree sobre las celdas ocupadas.

    En vez de precalcular la distancia al obstaculo mas cercano para TODAS las
    celdas (distance_transform_edt, O(celdas)), guardamos solo los obstaculos en
    un KD-tree y consultamos el vecino mas cercano por endpoint, O(log M) por rayo.
    Solo se paga por los rayos que realmente se evaluan.
    """

    def __init__(self, occupied_xy, sigma):
        self.sigma = sigma
        # cKDTree necesita al menos un punto; si no hay obstaculos, tree queda en None
        self.tree = cKDTree(occupied_xy) if occupied_xy.shape[0] > 0 else None

    def log_prob(self, points_xy):
        """
        points_xy: array (K, 2) con las coordenadas (x, y) en metros de los endpoints.
        Devuelve el log de la verosimilitud gaussiana de cada endpoint.
        """
        if self.tree is None:
            return np.full(points_xy.shape[0], LOG_MIN)

        # distancia (en metros) de cada endpoint al obstaculo ocupado mas cercano
        dist, _ = self.tree.query(points_xy)
        logp = -0.5 * (dist / self.sigma) ** 2  # log de la gaussiana
        return np.maximum(logp, LOG_MIN)


def precompute_likelihood_field(mapa, sigma):
    if hasattr(mapa, 'get_probability'):
        prob = mapa.get_probability()
        # Estricto: prob > 0.5 <=> log_odds > 0, o sea solo celdas con evidencia de ocupacion.
        # OJO: con >= se incluirian las celdas DESCONOCIDAS (log_odds = 0 -> prob = 0.5),
        # que son casi todo el mapa, llenando el likelihood field de obstaculos falsos.
        occupied = prob > 0.5  # mascara de celdas ocupadas
    else:
        occupied = mapa.occupied

    # Coordenadas (x, y) en metros de los centros de las celdas ocupadas
    rows, cols = np.where(occupied)
    xs = (cols + 0.5) * mapa.resolution + mapa.origin_x
    ys = (rows + 0.5) * mapa.resolution + mapa.origin_y
    occupied_xy = np.column_stack((xs, ys))

    return LikelihoodField(occupied_xy, sigma)


def compute_likelihood(pose, scan_ranges, scan_angles, range_min, field, subsample=2):
    x, y, theta = pose
    r = scan_ranges[::subsample]
    a = scan_angles[::subsample]
    valid = np.isfinite(r) & (r >= range_min)
    r, a = r[valid], a[valid]
    if r.size == 0:
        return 1e-6

    # Endpoints del scan en coordenadas del mundo (metros). Se consultan directo
    # contra el KD-tree, sin cuantizar a la grilla.
    ex = x + r * np.cos(theta + a)
    ey = y + r * np.sin(theta + a)
    points = np.column_stack((ex, ey))

    logp = field.log_prob(points)
    return np.exp(logp.mean())
