import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import os

# Settings
num_bins = 30
num_agents = 1000
num_plots = 100
output_dir = "beta_cdf_plots3"
os.makedirs(output_dir, exist_ok=True)

# Create uniform values
bin_edges = np.linspace(0, 10, num_bins + 1)
values = np.linspace(0, 10, num_agents)

# Plot generation loop
for i in range(num_plots):
    beta_a = np.random.uniform(1, 3)
    beta_b = np.random.uniform(0.5, 5)

    distorted_values = np.power(values, beta_a) # beta.cdf(values, beta_a, beta_b)
    unique_vals = np.linspace(0, 10, 1000)
    mapped_vals = np.power(unique_vals, beta_a)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Original Uniform Histogram
    axs[0].hist(values, bins=bin_edges, edgecolor='black')
    axs[0].set_title("Uniform Distribution")
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Frequency")

    # Plot 2: Distorted Histogram
    axs[1].hist(distorted_values, bins=bin_edges, edgecolor='black')
    axs[1].set_title(f"Distorted (Beta CDF)\na={beta_a:.2f}, b={beta_b:.2f}")
    axs[1].set_xlabel("Distorted Value")
    axs[1].set_ylabel("Frequency")

    # Plot 3: Mapping Curve
    axs[2].plot(unique_vals, unique_vals, '--', label='Identity', color='gray')
    axs[2].plot(unique_vals, mapped_vals, label='Beta CDF', color='blue')
    axs[2].set_title("Distortion Mapping")
    axs[2].set_xlabel("Original Value")
    axs[2].set_ylabel("Distorted Value")
    axs[2].legend()
    axs[2].grid(True)

    # Layout and Save
    plt.tight_layout()
    filename = f"{output_dir}/beta_cdf_triplet_{i+1:03d}_a{beta_a:.2f}_b{beta_b:.2f}.png"
    plt.savefig(filename)
    plt.close()

print(f"Saved {num_plots} 3-panel plots to {output_dir}/")
