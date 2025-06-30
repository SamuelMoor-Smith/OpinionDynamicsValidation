import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define the number of bins and values per bin
num_bins = 11
values_per_bin = 20

# Create uniform values: 20 of each step from 0.0 to 1.0
bin_edges = np.linspace(0, 1, num_bins + 1)
values = []
for x in range(11):
    values.extend([x / 10] * values_per_bin)
values = np.array(values)

# ---------- First Plot: Uniform Histogram ----------
plt.figure(figsize=(6, 4))
plt.hist(values, bins=bin_edges, edgecolor='black')
plt.title("Uniform Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.ylim(top=20)
plt.show()

# ---------- Second Plot: Histogram of Distorted Values ----------
# Define Beta distribution parameters
beta_a, beta_b = 1.25, 0.75
distorted_values = beta.cdf(values, beta_a, beta_b)

plt.figure(figsize=(6, 4))
plt.hist(distorted_values, bins=bin_edges, edgecolor='black')
plt.title(f"Histogram of Distorted Values (Beta CDF, a={beta_a}, b={beta_b})")
plt.xlabel("Distorted Value")
plt.ylabel("Frequency")
plt.ylim(top=20)
plt.show()

# ---------- Third Plot: Distortion Mapping ----------
unique_vals = np.linspace(0, 1, 1000)
mapped_vals = beta.cdf(unique_vals, beta_a, beta_b)

plt.figure(figsize=(6, 4))
plt.plot(unique_vals, unique_vals, '--', label='Identity (no distortion)', color='gray')
plt.plot(unique_vals, mapped_vals, label=f'Beta CDF (a={beta_a}, b={beta_b})')
plt.title("Distortion Mapping (Original → Distorted)")
plt.xlabel("Original Value")
plt.ylabel("Distorted Value")
plt.grid(True)
plt.legend()
plt.show()
