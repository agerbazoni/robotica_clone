import numpy as np

def pure_pursuit(pose, path, lookahead_dist, v_max=0.18, align_threshold=0.7):
    rx, ry, rtheta = pose
    path = np.array(path)

    # 1. Punto mas cercano del path al robot
    dists = np.hypot(path[:, 0] - rx, path[:, 1] - ry)
    closest_idx = np.argmin(dists)

    # 2. Caminar sobre el path acumulando distancia hasta completar L
    goal_idx = closest_idx
    accumulated = 0.0
    for i in range(closest_idx, len(path) - 1):
        seg = np.hypot(path[i+1, 0] - path[i, 0], path[i+1, 1] - path[i, 1])
        accumulated += seg
        if accumulated >= lookahead_dist:
            goal_idx = i + 1
            break
    else:
        goal_idx = len(path) - 1

    gx, gy = path[goal_idx]

    dx = gx - rx
    dy = gy - ry

    # transformar a coordenadas locales del robot
    x_local = np.cos(rtheta) * dx + np.sin(rtheta) * dy
    y_local = -np.sin(rtheta) * dx + np.cos(rtheta) * dy

    L_sq = x_local**2 + y_local**2
    if L_sq < 1e-6:
        return 0.0, 0.0, goal_idx

    gamma = 2.0 * y_local / L_sq

    # Error de rumbo hacia el punto de lookahead, en el marco del robot.
    alpha = np.arctan2(y_local, x_local)

    # Si el objetivo está muy desalineado con el rumbo actual (junta entre dos caminos
    # de waypoints consecutivos, donde A* ignora la orientación con la que llega el robot
    # y arranca en otra dirección), pivotamos en el lugar hasta encarar antes de avanzar.
    # Así evitamos el arco ancho de "girar y avanzar a la vez", que en un quiebro cerrado
    # cerca de un obstáculo termina clipeándolo. Incluye el caso del punto por detrás.
    if abs(alpha) > align_threshold:
        omega = np.clip(2.0 * alpha, -0.8, 0.8)
        return 0.0, omega, goal_idx

    v = v_max / (1.0 + 0.3 * abs(gamma))
    omega = v * gamma

    return v, omega, goal_idx