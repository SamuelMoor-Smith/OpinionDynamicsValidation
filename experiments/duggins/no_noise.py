import numpy as np
from utils.rand_gen import create_random_opinion_distribution
from models.duggins import DugginsModel
from datasets.dataset import Dataset
from utils.differences import calculate_mean_std
from utils.plotting import plot_2_datasets_snapshots
from utils import optimizers
import time
from utils.logging import write_results_to_file

# Create a model with random parameters
base_model = DugginsModel()
print(f"{DugginsModel} model created with random parameters: ", base_model.params)

# generate random initial opinions
op_range = base_model.get_opinion_range()
initial_opinions = create_random_opinion_distribution(N=1000, min_val=op_range[0], max_val=op_range[1])

# create agents for duggins model
base_model.create_agents(initial_opinions)

# Run this model for 10 steps and create dataset
start = time.time()
true = Dataset.create_with_duggins_model(base_model, num_steps=9)
print(f"Dataset creation took {time.time() - start} seconds")

start = time.time()
# Now create 10 more datasets with the same model and initial opinions
new_models = [DugginsModel(base_model.params) for _ in range(10)]
for model in new_models:
    model.create_agents(initial_opinions)
datasets = [Dataset.create_with_duggins_model(model, num_steps=9) for model in new_models]
print(f"10 datasets creation took {time.time() - start} seconds")

# Calculate mean and std of differences between the first dataset and the rest
base_mean_diff, base_std_diff = calculate_mean_std(true, datasets, "Baseline", method="wasserstein")

# Plot the true dataset and the first of the rest
plot_2_datasets_snapshots(true, datasets[0], difference="wasserstein", path=f"plots/duggins/no_noise/same/")

# Optimization process and time it
start = time.time()
comparison_model = DugginsModel()
optimizer = optimizers.get_optimizer()
best_params = optimizer(true, comparison_model, optimizers.hyperopt_objective)
print(f"Optimization took {time.time() - start} seconds")

# Set the best parameters
comparison_model.set_normalized_params(best_params)

# Print both params
print("Baseline model params: ", base_model.params)
print("Optimized model params: ", comparison_model.params)

# Now create 10 more datasets with the optimized model and initial opinions
# opt_datasets = [Dataset.create_with_model_from_initial(comparison_model, initial_opinions, num_steps=9) for _ in range(10)]
# Now create 10 more datasets with the same model and initial opinions
opt_models = [DugginsModel(comparison_model.params) for _ in range(10)]
for model in opt_models:
    model.create_agents(initial_opinions)
opt_datasets = [Dataset.create_with_duggins_model(model, num_steps=9) for model in opt_models]

# Calculate mean and std of differences between the first dataset and the rest
opt_mean_diff, opt_std_diff = calculate_mean_std(true, opt_datasets, "Optimized", method="wasserstein")

# Plot the true dataset and the first of the optimized
plot_2_datasets_snapshots(true, opt_datasets[0], difference="wasserstein", path=f"plots/duggins/no_noise/diff/")

# Write the results to a file
write_results_to_file(
    base_model.params, best_params, 
    base_mean_diff, base_std_diff, 
    opt_mean_diff, opt_std_diff, 
    path=f"results/duggins/no_noise/"
)

