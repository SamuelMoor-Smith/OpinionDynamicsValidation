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
    (0.3, 0.1),
    (0.5, 0.1)
]
colors = ['C2', 'C2']

output_dir = "gaussian_cdf_figures"
os.makedirs(output_dir, exist_ok=True)

for i in range(2):  # Loop for a single plot

    mu, sigma = params[i]

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
    # distorted_values = beta.cdf(values, beta_a, beta_b)
    from scipy.stats import norm
    # distorted_values = norm.cdf(values, loc=mu, scale=sigma)
    distorted_values = np.random.normal(loc=mu, scale=sigma, size=100000)

    plt.figure(figsize=(6, 4))
    plt.hist(distorted_values, bins=bin_edges, edgecolor='black', color=colors[i])
    plt.title(f"Histogram of Distorted Values (Beta CDF, a={mu}, b={sigma})")
    plt.xlabel("Distorted Value")
    plt.ylabel("Frequency")
    # plt.ylim(top=20)
    plt.savefig(f"{output_dir}/beta_cdf_histogram_a{mu}_b{sigma}.png")
    plt.close()

    # # ---------- Third Plot: Distortion Mapping ----------
    # unique_vals = np.linspace(0, 1, 1000)
    # mapped_vals = beta.cdf(unique_vals, beta_a, sigma)

    # plt.figure(figsize=(6, 4))
    # plt.plot(unique_vals, unique_vals, '--', label='Identity (no distortion)', color='gray')
    # plt.plot(unique_vals, mapped_vals, label=f'Beta CDF (a={beta_a}, b={beta_b})', color=colors[i])
    # plt.title(f"Distortion Mapping (Beta CDF, a={beta_a}, b={beta_b})")
    # plt.xlabel("Original Value")
    # plt.ylabel("Distorted Value")
    # plt.grid(True)
    # plt.legend()
    plt.savefig(f"{output_dir}/gaussian_mapping{mu}_b{sigma}.png")
    plt.close()
