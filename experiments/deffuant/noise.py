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

# noises = np.linspace(0, 0.5, 50)
# explained_variances = []
# for noise in noises:
#     explained_variance = 0
#     for _ in range(10):
#         true, basic_var, noisy_var = Dataset.create_with_model_from_initial_with_noise(base_model, initial_opinions, num_steps=9, noise=noise)
#         explained_variance += basic_var / noisy_var
#     explained_variances.append(explained_variance / 10)
#     print(f"Explained variance for noise {noise}: {basic_var / noisy_var}")

# plt.plot(noises, explained_variances)
# plt.xlabel("Noise level")
# plt.ylabel("Explained variance")
# plt.title("Explained variance for different noise levels")
# plt.savefig("plots/deffuant/noise/explained_variance.png")
    


# # Create zero data
# zero_data = copy.copy(true.get_data())
# zero_data.pop()
# zero_data.insert(0, zero_data[0])

# zero = Dataset.create_from_data(zero_data)

# zero_diff = dataset_difference(true, zero, method="wasserstein")
# print("Zero mean diff: ", zero_diff)

# # plot_2_snapshots(true.get_data()[0], zero.get_data()[0])

# plot_2_datasets_snapshots(true, zero, difference="wasserstein", path="plots/deffuant/noise/zero/")

# # Time the optimization process
# start = time.time()
# comparison_model = DeffuantModel()
# optimizer = optimizers.get_optimizer()
# best_params = optimizer(true, comparison_model, obj_f=optimizers.hyperopt_objective_noisy)
# print(f"Optimization took {time.time() - start} seconds")

# # Set the best parameters
# comparison_model.set_normalized_params(best_params)

# # Print both params
# print("Baseline model params: ", base_model.params)
# print("Optimized model params: ", comparison_model.params)

# # Now create 10 more datasets with the same model and initial opinions
# opt_datasets = [Dataset.create_with_model_from_true(comparison_model, true.get_data()) for _ in range(10)]

# # Calculate mean and std of differences between the first dataset and the rest
# opt_mean_diff, opt_std_diff = calculate_mean_std(true, opt_datasets, "Optimized", method="wasserstein")

# plot_2_datasets_snapshots(true, opt_datasets[0], difference="wasserstein", path="plots/deffuant/noise/diff/")

# # Write the results to a file
# write_results_to_file(
#     base_model.params, best_params, 
#     zero_diff, 0, 
#     opt_mean_diff, opt_std_diff, 
#     path="results/deffuant/noise/"
# )
