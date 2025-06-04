import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
import pandas as pd

# Parameters
R_large = 523.4  # Radius of the large circle in meters
R_small = 214.5  # Radius of the small circles in meters
D_large = 2 * R_large  # Diameter of the large circle

# Overlap configuration
overlap_factor = 0.85  # < 1 ensures overlap

# Spacing between circle centers with overlap
dx_overlap = 2 * R_small * overlap_factor  # Horizontal spacing
dy_overlap = R_small * np.sqrt(3) * overlap_factor  # Vertical spacing

# Number of rows
num_rows_overlap = ceil(D_large / dy_overlap) + 1

# Generate grid of circle centers
circle_centers_overlap = []

for j in range(num_rows_overlap):
    y = -R_large + j * dy_overlap
    row_offset = (j % 2) * (dx_overlap / 2)  # Stagger every other row
    num_cols = ceil(D_large / dx_overlap) + 1
    for i in range(num_cols):
        x = -R_large + row_offset + i * dx_overlap
        # Ensure the small circle covers part of the large circle
        if sqrt(x**2 + y**2) <= R_large + R_small:
            circle_centers_overlap.append((x, y))

# Convert to DataFrame
df_overlap_centers = pd.DataFrame(circle_centers_overlap, columns=["x", "y"])

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the large circle
large_circle = plt.Circle((0, 0), R_large, color='blue', fill=False, linewidth=2, label='Search Area')
ax.add_patch(large_circle)

# Plot the small circles
for idx, (x, y) in enumerate(circle_centers_overlap):
    # Only add label to the first small circle for legend
    if idx == 0:
        small_circle = plt.Circle((x, y), R_small, color='green', fill=False, linewidth=1, label='Laser Beam')
    else:
        small_circle = plt.Circle((x, y), R_small, color='green', fill=False, linewidth=1)
    ax.add_patch(small_circle)

# Display settings
ax.set_aspect('equal')
ax.set_xlim(-R_large - R_small, R_large + R_small)
ax.set_ylim(-R_large - R_small, R_large + R_small)
plt.grid(True)
ax.legend()
plt.show()
