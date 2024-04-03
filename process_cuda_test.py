import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

slice = [2,3]
infile = '/home/thomas/Documents/thesis/cuda-roa/initial_states.bin'
finfile = '/home/thomas/Documents/thesis/cuda-roa/final_states.bin'
# Read binary files
with open(infile, 'rb') as file:
    initial_states = np.fromfile(file, dtype=np.float64).reshape((-1, 8))
with open(finfile, 'rb') as file:
    final_states = np.fromfile(file, dtype=np.float64).reshape((-1, 8))

x0 = np.array([0,0,0,0,0,0,0,0])
stable = []
for _ in range(len(final_states)):
    atFixedPt = np.allclose(final_states[_, [2,3]], x0[[2,3]], atol=0.03) and np.allclose(final_states[_, [6,7]], x0[[6,7]], atol=0.1)
    # atFixedPt = np.allclose(final_states[_, [1,2]], x0[[1,2]], atol=0.01) and np.allclose(final_states[_, [4,5]], x0[[4,5]], atol=0.01)

    stable.append(atFixedPt)
stable = np.array(stable)

# Create scatter plot of final states
plt.figure(figsize=(8, 7))
plt.subplot(1, 2, 1)
plt.scatter(final_states[:, slice[0]], final_states[:, slice[1]], s=1)
plt.title(f'Final States ')
plt.xlabel('X' + str(slice[0]))
plt.ylabel('X' + str(slice[1]))
plt.xlim(-np.pi, np.pi)
plt.ylim(-10, 10)

# Create scatter plot of initial states that led to stability
plt.subplot(1, 2, 2)

plt.scatter(initial_states[stable, slice[0]], initial_states[stable, slice[1]], s=1, color='blue', label='Stable', alpha=0.1)
plt.scatter(initial_states[~stable, slice[0]], initial_states[~stable, slice[1]], s=1, color='red', label='Unstable', alpha=0.1)
plt.title(f'Region of Attraction (Num Stable: {np.sum(stable)}, Total: {len(final_states)})')
plt.xlabel('X' + str(slice[0]))
plt.ylabel('X' + str(slice[1]))

plt.tight_layout()
plt.savefig('roa.png')
plt.show()


# Define the grid over which you want to visualize the stability boundary
x_range = np.linspace(min(initial_states[:, slice[0]]), max(initial_states[:, slice[0]]), 100)
y_range = np.linspace(min(initial_states[:, slice[1]]), max(initial_states[:, slice[1]]), 100)
X, Y = np.meshgrid(x_range, y_range)

# Interpolate the stability information over this grid
# We assign a value of 1 for stable and 0 for unstable to interpolate between
stability_values = np.where(stable, 1, 0)
Z = griddata(initial_states[:, slice], stability_values, (X, Y), method='cubic')

plt.scatter(initial_states[stable, slice[0]], initial_states[stable, slice[1]], s=1, color='blue', label='Stable', alpha=0.1)
plt.scatter(initial_states[~stable, slice[0]], initial_states[~stable, slice[1]], s=1, color='red', label='Unstable', alpha=0.1)
# Use a contour plot to draw the boundary between stable (1) and unstable (0) states
contour = plt.contour(X, Y, Z, levels=[0.5], colors='k')  # Adjust levels for finer control

plt.title('Boundary Between Stable and Unstable States')
plt.xlabel('X' + str(slice[0]))
plt.ylabel('X' + str(slice[1]))
plt.show()