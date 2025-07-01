import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import graphx as gx # type: ignore

SAMPLES = 10000


points = np.zeros((SAMPLES, 2))
for i in range(SAMPLES):
    points[i] = gx.random_points_in_disk_with_attraction_point(np.array([0, 0]), 1, np.array([1.5, 0]), 0.5) # type: ignore

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