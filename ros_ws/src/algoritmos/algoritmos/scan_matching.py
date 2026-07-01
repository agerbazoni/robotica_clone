import numpy as np

# Scan matching local sobre el likelihood field (estilo FastSLAM 2.0 / gmapping).
#
# Dada una pose inicial (la prediccion de odometria de una particula), se busca
# la pose que maximiza la verosimilitud del scan contra el mapa, usando un
# hill-climbing con paso que se va afinando. El resultado se usa como centro de
# la propuesta, en vez de propagar solo el modelo de movimiento.


def make_beam_points(scan_ranges, scan_angles, range_min, subsample=10):
    """
    Precalcula los endpoints del scan en el marco del sensor (constantes para la
    busqueda, ya que no dependen de la pose). Devuelve (bx, by).

    Asi, al evaluar una pose candidata (x, y, theta) solo hace falta una rotacion
    2D, sin recalcular cos/sin por rayo en cada paso del hill-climbing.
    """
    r = np.asarray(scan_ranges)[::subsample]
    a = np.asarray(scan_angles)[::subsample]
    valid = np.isfinite(r) & (r >= range_min)
    r, a = r[valid], a[valid]
    bx = r * np.cos(a)
    by = r * np.sin(a)
    return bx, by


def _score(pose, bx, by, field):
    """Suma de log-verosimilitud de los endpoints del scan en la pose dada."""
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    ex = x + bx * c - by * s
    ey = y + bx * s + by * c
    points = np.column_stack((ex, ey))
    return field.log_prob(points).sum()


def scan_match(pose, bx, by, field, linear_step=0.1, angular_step=0.08, iters=4):
    """
    Refina 'pose' (x, y, theta) maximizando la verosimilitud del scan contra el
    likelihood field. Hill-climbing greedy: prueba moverse +/- un paso en x, y y
    theta; cuando ningun vecino mejora, reduce el paso a la mitad y repite.

    Devuelve la pose refinada (no muta la entrada).
    """
    best = np.array(pose, dtype=float)
    if bx.size == 0:
        return best

    best_score = _score(best, bx, by, field)

    # Direcciones de busqueda: +/- en cada eje (x, y, theta)
    moves = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float)

    lin, ang = linear_step, angular_step
    for _ in range(iters):
        improved = True
        while improved:
            improved = False
            for m in moves:
                cand = best + m * np.array([lin, lin, ang])
                s = _score(cand, bx, by, field)
                if s > best_score:
                    best_score, best, improved = s, cand, True
        # Afinar la resolucion de busqueda
        lin *= 0.5
        ang *= 0.5

    return best
