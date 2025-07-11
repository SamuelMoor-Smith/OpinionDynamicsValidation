import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import beta

# Define the number of bins and values per bin
num_bins = 50
num_agents = 1000

# Create uniform values: 20 of each step from 0.0 to 1.0
bin_edges = np.linspace(0, 1, num_bins + 1)
values = np.linspace(0, 1, num_agents)

params = [
    (2, 2),
    (0.5, 0.5),
    (1, 2),
    (2, 1),
    (1.5, 2),
    (0.8, 0.5)
]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

output_dir = "beta_cdf_figures"
os.makedirs(output_dir, exist_ok=True)

for i in range(6):  # Loop for a single plot

    beta_a, beta_b = params[i]

    # # ---------- First Plot: Uniform Histogram ----------
    # plt.figure(figsize=(6, 4))
    # plt.hist(values, bins=bin_edges, edgecolor='black')
    # plt.title("Uniform Distribution")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # # plt.ylim(top=20)
    # plt.show()

    # ---------- Second Plot: Histogram of Distorted Values ----------
    # Define Beta distribution parameters
    distorted_values = beta.cdf(values, beta_a, beta_b)

    plt.figure(figsize=(6, 4))
    plt.hist(distorted_values, bins=bin_edges, edgecolor='black', color=colors[i])
    plt.title(f"Histogram of Distorted Values (Beta CDF, a={beta_a}, b={beta_b})")
    plt.xlabel("Distorted Value")
    plt.ylabel("Frequency")
    # plt.ylim(top=20)
    plt.savefig(f"{output_dir}/beta_cdf_histogram_a{beta_a}_b{beta_b}.png")
    plt.close()

    # ---------- Third Plot: Distortion Mapping ----------
    unique_vals = np.linspace(0, 1, 1000)
    mapped_vals = beta.cdf(unique_vals, beta_a, beta_b)

    plt.figure(figsize=(6, 4))
    plt.plot(unique_vals, unique_vals, '--', label='Identity (no distortion)', color='gray')
    plt.plot(unique_vals, mapped_vals, label=f'Beta CDF (a={beta_a}, b={beta_b})', color=colors[i])
    plt.title(f"Distortion Mapping (Beta CDF, a={beta_a}, b={beta_b})")
    plt.xlabel("Original Value")
    plt.ylabel("Distorted Value")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/beta_cdf_mapping_a{beta_a}_b{beta_b}.png")
    plt.close()
