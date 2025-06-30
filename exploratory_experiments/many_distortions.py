import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Settings
num_bins = 50
values_per_bin = 10
num_plots = 50
output_dir = "beta_cdf_plots"

# Create uniform values
bin_edges = np.linspace(0, 1, num_bins + 1)
values = np.linspace(0, 1, num_bins * values_per_bin, endpoint=False)

# Create output directory if it doesn't exist
import os
os.makedirs(output_dir, exist_ok=True)

# Generate and save plots
for i in range(num_plots):
    beta_a = np.random.uniform(0.5, 5)  # Avoid a, b < 0.5 to reduce extremes
    beta_b = np.random.uniform(0.5, 5)

    distorted_values = beta.cdf(values, beta_a, beta_b)

    plt.figure(figsize=(6, 4))
    plt.hist(distorted_values, bins=bin_edges, edgecolor='black')
    plt.title(f"Beta CDF Histogram (a={beta_a:.2f}, b={beta_b:.2f})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    # Save the plot
    filename = f"{output_dir}/beta_cdf_plot_{i+1:02d}_a{beta_a:.2f}_b{beta_b:.2f}.png"
    plt.savefig(filename)
    plt.close()  # Close the figure to save memory

print(f"Saved {num_plots} plots to {output_dir}/")