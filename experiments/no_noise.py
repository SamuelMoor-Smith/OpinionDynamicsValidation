from datasets.dataset import Dataset
from utils.differences import calculate_mean_std
from utils.plotting import plot_2_datasets_snapshots
from utils import optimizers
from models.model import Model
from models.duggins import DugginsModel
import time
from utils.my_logging import write_results_to_file
from utils.differences import dataset_difference, dataset_difference_early
import os
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os

def save_experiment_results(filename, data):
    """
    Save experiment results to a CSV file.
    
    Parameters:
    filename (str): Path to the CSV file.
    data (dict): Dictionary containing experiment data with keys as column names.
    """
    df = pd.DataFrame(data)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"Results saved to {filename}")

# Example usage:
# data = {
#     "noise": [0.1, 0.2, 0.3],
#     "zero_diff": [0.5, 0.6, 0.7],
#     "explained_variance": [0.9, 0.85, 0.8],
#     "opt_mean_diff": [0.4, 0.35, 0.3],
#     "opt_std_diff": [0.05, 0.06, 0.07]
# }
# save_experiment_results("results/experiment_results.csv", data)

def no_noise_experiment(
        model_class: Model,
        i=""
    ):

    model_name = model_class.get_model_name()

    # Create the array of explained variances and score differences
    explained_variances = []    
    score_diffs = []
    base_score_diffs = []
    opt_mean_diffs = []
    opt_std_diffs = []
    zero_diffs = []
    base_mean_diffs = []
    base_std_diffs = []

    # Create non-uniformly sampled noise levels
    noises = np.linspace(0, 0, 100)  # Uniformly spaced values between 0 and 1
    # noises = 0.5 * uniform_samples**2  # Square root transformation to bias towards smaller values
    
    print(f"Noises: {noises}")

    for noise in noises:

        base_model: Model = model_class() # Create model with random parameters
        initial_opinions = base_model.generate_initial_opinions() # generate random initial opinions

        if isinstance(base_model, DugginsModel):
            base_model.sample_isc_for_agents(initial_opinions)

        # Iterate over the noise levels and create the true data with that noise value
        true, explained_var = Dataset.create_with_model_from_initial_with_noise(base_model, initial_opinions, num_steps=9, noise=noise)

        # # Now create 10 more datasets with the base model and initial opinions
        # base_datasets = [Dataset.create_with_model_from_true(base_model, true.get_data()) for _ in range(10)]

        # # Calculate mean and std of differences between the first dataset and the rest
        # base_mean_diff, base_std_diff = calculate_mean_std(true, base_datasets, "Baseline", method="wasserstein")
        # base_mean_diffs.append(base_mean_diff)
        # base_std_diffs.append(base_std_diff)

        # if explained_var < 0.05:
        #     continue

        # Create zero data (just the last opinion to predict the next one)
        zero = Dataset.create_zero_data_from_true(true, base_model)

        # Calculate the difference between the true and zero datasets
        zero_diff = dataset_difference(true, zero, method="wasserstein")
        zero_diffs.append(zero_diff)

        if isinstance(base_model, DugginsModel):
            comparison_model: Model = DugginsModel(agents=base_model.get_cleaned_agents())
        else:
            comparison_model: Model = model_class()

        # Optimization process and time it
        start = time.time()
        optimizer = optimizers.get_optimizer()
        opt_params = {"from_true": True, "num_snapshots": 10}
        best_params = optimizer(true, comparison_model, opt_params, obj_f=optimizers.hyperopt_objective)
        print(f"Optimization took {time.time() - start} seconds")

        # Set the best parameters
        comparison_model.set_normalized_params(best_params)
        print("Best parameters: ", comparison_model.params)

        # Now create 10 more datasets with the optimized model and initial opinions
        opt_datasets = [Dataset.create_with_model_from_true(comparison_model, true.get_data()) for _ in range(10)]

        # Calculate mean and std of differences between the first dataset and the rest
        opt_mean_diff, opt_std_diff = calculate_mean_std(true, opt_datasets, "Optimized", method="wasserstein")
        opt_mean_diffs.append(opt_mean_diff)
        opt_std_diffs.append(opt_std_diff)

        # Print and store the score difference between the zero and optimized datasets
        print(f"Score difference for noise {noise}: {zero_diff - opt_mean_diff}")
        explained_variances.append(explained_var)
        score_diffs.append((zero_diff - opt_mean_diff)/zero_diff)
        # base_score_diffs.append((zero_diff - base_mean_diff)/zero_diff)

        # plot_2_datasets_snapshots(true, zero, difference="wasserstein", path=f"plots/{model_name}/noise/")

    if not os.path.exists(f"results/{model_name}/noise"):
        os.makedirs(f"results/{model_name}/noise")

    filename = f"results/{model_name}/noise/no_noise-varying_noise_data_{i}.csv"
    data = {
        "noise": noises,
        "zero_diff": zero_diffs,
        "explained_variance": explained_variances,
        "opt_mean_diff": opt_mean_diffs,
        "opt_std_diff": opt_std_diffs
    }

    save_experiment_results(filename, data)
    # Plot the explained variances
    plt.scatter(noises, score_diffs)
    # plt.scatter(noises, base_score_diffs)
    plt.xlabel("Noise Level")
    plt.ylabel("Score difference")
    plt.title("Score difference for different explained variance levels")
    plt.savefig(f"plots/{model_name}/noise/no_noise-score_diff_explained_var_{i}.png")
    plt.close()