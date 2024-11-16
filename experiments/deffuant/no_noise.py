import numpy as np
from utils.rand_gen import create_random_opinion_distribution
from models.deffuant import DeffuantModel
from datasets.dataset import Dataset
from utils.differences import dataset_difference
from utils.plotting import plot_2_datasets_snapshots

# generate random initial opinions
initial_opinions = create_random_opinion_distribution(N=1000, min_val=0, max_val=1, seed=42)

# Create a model with random parameters
base_model = DeffuantModel()
print("Model created with random parameters: ", base_model.params)

# Run this model for 10 steps and create dataset
true = Dataset.create(base_model, initial_opinions, num_steps=9)

# Now create 10 more datasets with the same model and initial opinions
datasets = [Dataset.create(base_model, initial_opinions, num_steps=9) for _ in range(10)]

# Calculate mean and std of differences between the first dataset and the rest
diffs = [dataset_difference(true, d, method="wasserstein") for d in datasets]
mean_diff = np.mean(diffs)
std_diff = np.std(diffs)

print(f"Mean difference: {mean_diff}, Std difference: {std_diff}")

plot_2_datasets_snapshots(true, datasets[0], difference="wasserstein", path="plots/deffuant/no_noise/")
