import numpy as np


def effective_sample_size(weights):
    """
    Calcula el tamaño efectivo de muestra (N_eff) a partir de los pesos normalizados.
    Si N_eff < N/2, conviene resamplear.
    """
    return 1.0 / np.sum(weights ** 2)

def sus(particles, weights, maps=None):
    N = len(weights)
    new_particles = np.zeros_like(particles)
    new_maps = [] if maps is not None else None

    initial_offset = np.random.uniform(0, 1.0 / N)
    weight = weights[0]
    idx = 0

    for i in range(N):
        u = initial_offset + i / N
        while u > weight:
            idx += 1
            weight += weights[idx]
        new_particles[i] = particles[idx]
        if maps is not None:
            new_maps.append(maps[idx].copy())

    new_weights = np.ones(N) / N

    if maps is not None:
        return new_particles, new_weights, new_maps
    return new_particles, new_weights

def redistribute_uniform(map_data, N):
    free_rows, free_cols = np.where(map_data.free)
    indices = np.random.choice(len(free_rows), size=N)

    particles = np.zeros((N, 3))
    particles[:, 0], particles[:, 1] = map_data.grid_to_world(free_cols[indices], free_rows[indices])
    particles[:, 2] = np.random.uniform(-np.pi, np.pi, N)

    return particles

def get_selected_state(particles, weights):
    mean_x = np.dot(weights, particles[:, 0])
    mean_y = np.dot(weights, particles[:, 1])

    mean_theta = np.arctan2(
        np.dot(weights, np.sin(particles[:, 2])),
        np.dot(weights, np.cos(particles[:, 2])))

    return [mean_x, mean_y, mean_theta]