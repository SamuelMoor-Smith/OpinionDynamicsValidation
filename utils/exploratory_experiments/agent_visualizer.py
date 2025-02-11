import numpy as np
import matplotlib.pyplot as plt

# Set parameters
num_agents = 1000
grid_size = 1000
radius_mean = 22
radius_std = 4

# Generate random agent positions
x_positions = np.random.uniform(0, grid_size, num_agents)
y_positions = np.random.uniform(0, grid_size, num_agents)

# Generate radii from normal distribution
radii = np.random.normal(radius_mean, radius_std, num_agents)
radii = np.clip(radii, 1, None)  # Ensure no negative or zero radii

# Create plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal')

# Plot agents as points
ax.scatter(x_positions, y_positions, color='red', s=5, label='Agents')

# Plot circles
for x, y, r in zip(x_positions, y_positions, radii):
    circle = plt.Circle((x, y), r, color='b', fill=False, alpha=0.3)
    ax.add_patch(circle)

# Show the plot
plt.title("Agent Visualization with Radii")
plt.show()
