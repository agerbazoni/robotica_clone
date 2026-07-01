import numpy as np

def sample_motion_model_odometry(particles, odom, noise):
    """
    particles: array de dimensión (N, 3)
    """
    N = particles.shape[0]

    delta_t = odom['t']       
    delta_r1 = odom['r1']
    delta_r2 = odom['r2']

    a1, a2, a3, a4 = noise

    sigma_r1 = np.sqrt(a1 * delta_r1 ** 2 + a2 * delta_t ** 2)
    sigma_r2 = np.sqrt(a1 * delta_r2 ** 2 + a2 * delta_t ** 2)
    sigma_t  = np.sqrt(a3 * delta_t ** 2  + a4 * (delta_r1 ** 2 + delta_r2 ** 2))

    delta_r1_hat = delta_r1 + np.random.normal(0, sigma_r1, N)
    delta_r2_hat = delta_r2 + np.random.normal(0, sigma_r2, N)
    delta_t_hat = delta_t + np.random.normal(0, sigma_t, N)

    particles[:, 0] += delta_t_hat * np.cos(particles[:, 2] + delta_r1_hat)
    particles[:, 1] += delta_t_hat * np.sin(particles[:, 2] + delta_r1_hat)
    particles[:, 2] = (particles[:, 2] + delta_r1_hat + delta_r2_hat + np.pi) % (2 * np.pi) - np.pi #normalizacion

    return particles