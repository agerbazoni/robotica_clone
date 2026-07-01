import numpy as np
from scipy.ndimage import distance_transform_edt

def precompute_likelihood_field(mapa, sigma, resolution=None):
    if hasattr(mapa, 'occupied'):
        occupied = mapa.occupied
        resolution = mapa.resolution
    else:
        occupied = mapa

    free = ~occupied
    if not np.any(occupied):
        return np.ones(occupied.shape) * 1e-6

    dist_map = distance_transform_edt(free) * resolution # distancia en celdas de cada celda libre a la celda ocupada más cercana (en metros)

    likelihood = np.exp(-0.5 * (dist_map / sigma) ** 2)

    return likelihood


def compute_likelihood(pose, scan_ranges, scan_angles, range_min, likelihood_field, mapa, subsample = 5):
    x, y, theta = pose
   
    log_w = 0.0
    count = 0

    for i in range(0, len(scan_ranges), subsample):
        r = scan_ranges[i]
        angle = scan_angles[i]
       
        if not np.isfinite(r) or r < range_min:
            continue
       
        endpoint_x = x + r * np.cos(theta + angle)
        endpoint_y = y + r * np.sin(theta + angle)

        col, row = mapa.world_to_grid(endpoint_x, endpoint_y)

        if 0 <= row < mapa.height and 0 <= col < mapa.width:
            prob = max(likelihood_field[row, col], 1e-6)
        else:
            prob = 1e-6
       
        log_w += np.log(prob)
        count += 1

    if count == 0:
        return 1e-6
    
    return np.exp(log_w / count)