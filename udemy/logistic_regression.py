import numpy as np
import matplotlib.pyplot as plt

n_pts = 100
np.random.seed(0)
random_x1_values = np.random.normal(10, 2, n_pts)
random_y1_values = np.random.normal(12, 2, n_pts)
top_region = np.array([random_x1_values, random_y1_values]).transpose() # just .T

random_x2_values = np.random.normal(5, 2, n_pts)
random_y2_values = np.random.normal(6, 2, n_pts)
bottom_region = np.array([random_x2_values, random_y2_values]).transpose() # just .T

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color = 'r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color = 'b')
plt.show()