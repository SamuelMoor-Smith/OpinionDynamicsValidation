import numpy as np
from utils.rand_gen import create_random_opinion_distribution
from models.deffuant import DeffuantModel
from datasets.dataset import Dataset
from utils.differences import calculate_mean_std, dataset_difference
from utils.plotting import plot_2_datasets_snapshots, plot_2_snapshots
from utils import optimizers
import time
from utils.logging import write_results_to_file
import copy
import matplotlib.pyplot as plt

# generate random initial opinions
initial_opinions = create_random_opinion_distribution(N=1000, min_val=0, max_val=1, seed=42)

# Create a model with random parameters
base_model = DeffuantModel()
print("Model created with random parameters: ", base_model.params)

# # Run this model with noise for 10 steps and create dataset
true, _, _ = Dataset.create_with_model_from_initial_with_noise(base_model, initial_opinions, num_steps=9, noise=0.15)

# Create the array of noise levels and of explained variances
noises = np.linspace(0, 0.5, 50)
explained_variances = []

# Iterate over the noise levels and calculate the average explained variance over 10 runs
for noise in noises:
    explained_variance = 0
    for _ in range(10):
        true, basic_var, noisy_var = Dataset.create_with_model_from_initial_with_noise(base_model, initial_opinions, num_steps=9, noise=noise)
        explained_variance += basic_var / noisy_var
    explained_variances.append(explained_variance / 10)
    print(f"Explained variance for noise {noise}: {basic_var / noisy_var}")

# Plot the explained variances
plt.plot(noises, explained_variances)
plt.xlabel("Noise level")
plt.ylabel("Explained variance")
plt.title("Explained variance for different noise levels")
plt.savefig("plots/deffuant/noise/explained_variance.png")
