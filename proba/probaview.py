import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SAMPLES = 1000000

def random_points_on_circle_with_attraction_point(circle_center, radius, attract_point, beta_1=2, beta_2=2, n_samples=1000):
    # Génère tous les dist et angles d'un coup
    # dist = 1-np.random.beta(1, beta_1, size=n_samples)
    dist = np.random.uniform(0, 1, size=n_samples)**0.5
    dist = np.sqrt(dist) * radius

    angles = np.random.uniform(-np.pi, np.pi, size=n_samples)
    angle_attract = np.arctan2(attract_point[1] - circle_center[1], attract_point[0] - circle_center[0])
    # alpha = 1-np.random.beta(1, beta_2, size=n_samples)
    alpha = np.random.uniform(0, 1, size=n_samples) ** 0.5
    angles = (1 - alpha) * angles + alpha * angle_attract
    angles = angles % (2*np.pi)

    x = circle_center[0] + dist * np.cos(angles)
    y = circle_center[1] + dist * np.sin(angles)
    return np.column_stack((x, y))

points = random_points_on_circle_with_attraction_point(
    np.array([0, 0]), 1, np.array([0.5, 0.5]), 1.1, 1.1, SAMPLES # type: ignore
)

plt.figure(figsize=(10, 10))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal', adjustable='box')

# Affichage de la densité
plt.hexbin(points[:, 0], points[:, 1], gridsize=200, cmap='inferno', extent=(-1.5, 1.5, -1.5, 1.5))
plt.colorbar(label='Densité de points')
circle = plt.Circle((0, 0), 1, color='r', fill=False, linewidth=2) # type: ignore
plt.gca().add_patch(circle)
plt.show()