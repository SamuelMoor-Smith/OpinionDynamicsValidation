from datasets.dataset import Dataset
from utils.differences import calculate_mean_std
from utils.plotting import plot_2_datasets_snapshots
from utils import optimizers
from models.model import Model
from models.duggins import DugginsModel
import time
from utils.logging import write_results_to_file
from utils.differences import dataset_difference
import os
import matplotlib.pyplot as plt

def varying_noise_experiment(
        model_class: Model,
        model_name: str,
        i=""
    ):

    base_model: Model = model_class() # Create model with random parameters
    initial_opinions = base_model.generate_initial_opinions() # generate random initial opinions

    if isinstance(base_model, DugginsModel):
        base_model.sample_isc_for_agents(initial_opinions)

    # Create the array of explained variances and score differences
    explained_variances = []
    score_diffs = []
    noises = [0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55]
    
    for noise in noises:

        # Iterate over the noise levels and create the true data with that noise value
        true, explained_var = Dataset.create_with_model_from_initial_with_noise(base_model, initial_opinions, num_steps=9, noise=noise)

        # Create zero data (just the last opinion to predict the next one)
        zero = Dataset.create_zero_data_from_true(true, base_model)

        # Calculate the difference between the true and zero datasets
        zero_diff = dataset_difference(true, zero, method="wasserstein")

        # Optimization process and time it
        start = time.time()
        comparison_model: Model = model_class()
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

        # Print and store the score difference between the zero and optimized datasets
        print(f"Score difference for noise {noise}: {zero_diff - opt_mean_diff}")
        explained_variances.append(explained_var)
        score_diffs.append(zero_diff - opt_mean_diff)

        # plot_2_datasets_snapshots(true, zero, difference="wasserstein", path=f"plots/{model_name}/noise/")

    if not os.path.exists(f"plots/{model_name}/noise"):
        os.makedirs(f"plots/{model_name}/noise")

    # Plot the explained variances
    plt.scatter(explained_variances, score_diffs)
    plt.xlabel("Explained Variance")
    plt.ylabel("Score difference")
    plt.title("Score difference for different explained variance levels")
    plt.savefig(f"plots/{model_name}/noise/score_diff_explained_var_{i}.png")
    plt.close()