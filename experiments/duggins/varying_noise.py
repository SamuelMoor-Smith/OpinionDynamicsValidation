import numpy as np
from utils.rand_gen import create_random_opinion_distribution
from datasets.dataset import Dataset
from models.duggins import DugginsModel
from utils.differences import calculate_mean_std, dataset_difference
from utils.plotting import plot_2_datasets_snapshots, plot_2_snapshots
from utils import optimizers
import time
from utils.logging import write_results_to_file
import copy
import matplotlib.pyplot as plt
import os

# Create a model with random parameters
base_model = DugginsModel()
print(f"{DugginsModel} model created with random parameters: ", base_model.params)

# generate random initial opinions
op_range = base_model.get_opinion_range()
initial_opinions = create_random_opinion_distribution(N=1000, min_val=op_range[0], max_val=op_range[1])

# Sample positions in bulk
x_positions = np.random.uniform(0, 1000, 1000)
y_positions = np.random.uniform(0, 1000, 1000)

# create agents for duggins model
base_model.set_positions(x_positions, y_positions)
agents = base_model.create_agents(initial_opinions)

# Create the array of explained variances and score differences
explained_variances = []
score_diffs = []
noises = [0, 0.04, 0.08, 0.1, 0.15, 0.25]

for noise in noises:

    # Iterate over the noise levels and create the true data with that noise value
    start = time.time()
    true, explained_var = Dataset.create_with_model_from_initial_with_noise(base_model, initial_opinions, num_steps=9, noise=noise)
    print(f"Dataset creation took {time.time() - start} seconds")

    # Create zero data (just the last opinion to predict the next one)
    zero_data = copy.copy(true.get_data())
    zero_data.pop()
    zero_data.insert(0, zero_data[0])
    zero = Dataset.create_from_data(zero_data, base_model)

    # Calculate the difference between the true and zero datasets
    zero_diff = dataset_difference(true, zero, method="wasserstein")
    print(f"Zero difference for noise {noise}: {zero_diff}")

    # Optimization process and time it
    start = time.time()
    comparison_model = DugginsModel()
    comparison_model.update_agents(agents, initial_opinions)
    optimizer = optimizers.get_optimizer()
    opt_params = {"from_true": False, "num_snapshots": 10}
    best_params = optimizer(true, comparison_model, opt_params, obj_f=optimizers.hyperopt_objective_from_true)
    print(f"Optimization took {time.time() - start} seconds")

    # Set the best parameters
    comparison_model.set_normalized_params(best_params)
    print("Best parameters: ", comparison_model.params)

    # Now create 10 more datasets with the optimized model and initial opinions
    # opt_datasets = [Dataset.create_with_model_from_true(comparison_model, true.get_data()) for _ in range(10)]
    opt_models = [DugginsModel(comparison_model.params) for _ in range(1)]
    for model in opt_models:
        model.update_agents(agents, initial_opinions)
    opt_datasets = [Dataset.create_with_model_from_true(model, true.get_data()) for model in opt_models]

    # Calculate mean and std of differences between the first dataset and the rest
    opt_mean_diff, opt_std_diff = calculate_mean_std(true, opt_datasets, "Optimized", method="wasserstein")

    # Print and store the score difference between the zero and optimized datasets
    print(f"Score difference for noise {noise}: {zero_diff - opt_mean_diff}")
    explained_variances.append(explained_var)
    score_diffs.append(zero_diff - opt_mean_diff)

    # Plot the true dataset and the first of the rest
    # plot_2_datasets_snapshots(true, zero, difference="wasserstein", path=f"plots/duggins/zero/")
    # plot_2_datasets_snapshots(true, opt_datasets[0], difference="wasserstein", path=f"plots/duggins/nonzero/")

if not os.path.exists(f"plots/duggins/noise"):
    os.makedirs(f"plots/duggins/noise")

# Plot the explained variances
plt.scatter(explained_variances, score_diffs)
plt.xlabel("Explained Variance")
plt.ylabel("Score difference")
plt.title("Score difference for different explained variance levels")
plt.savefig(f"plots/duggins/noise/score_diff_explained_var.png")
plt.close()