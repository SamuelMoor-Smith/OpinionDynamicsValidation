import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define x range
x = np.linspace(0, 1, 1000)

# Define (alpha, beta) parameter pairs
params = [
    (2, 2),
    (0.5, 0.5),
    (1, 2),
    (2, 1),
    (1.5, 2),
    (0.8, 0.5)
]

# Corresponding labels and colors
labels = [
    r'$\alpha = \beta = 2.0$',
    r'$\alpha = \beta = 0.5$',
    r'$\alpha = 1.0, \beta = 2.0$',
    r'$\alpha = 2.0, \beta = 1.0$',
    r'$\alpha = 1.5, \beta = 2.0$',
    r'$\alpha = 0.8, \beta = 0.5$'
]

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

# Plot
plt.figure(figsize=(8, 6))

plt.plot(x, x, '--', color='gray', label='No Distortion (y = x)')
for (a, b), label, color in zip(params, labels, colors):
    plt.plot(x, beta.cdf(x, a, b), label=label, color=color)

plt.title("Distortion Mapping with Beta CDF", fontsize=18, weight='bold')
plt.xlabel("Ordinal Opinion Value", fontsize=16)
plt.ylabel("Distorted Opinion Value", fontsize=16)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("beta_cdf_plot.png")
