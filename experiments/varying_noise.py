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
import os

def varying_noise_experiment(
        model_class,
        model_type,
        i=""
    ):

    # Create a model with random parameters
    base_model = model_class()
    print(f"{model_class} model created with random parameters: ", base_model.params)

    # generate random initial opinions
    op_range = base_model.get_opinion_range()
    initial_opinions = create_random_opinion_distribution(N=1000, min_val=op_range[0], max_val=op_range[1])

    # Create the array of explained variances and score differences
    explained_variances = []
    score_diffs = []
    noises = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.25, 0.3]
    
    for noise in noises:

        # Iterate over the noise levels and create the true data with that noise value
        true, basic_var, noisy_var = Dataset.create_with_model_from_initial_with_noise(base_model, initial_opinions, num_steps=9, noise=noise)

        # Create zero data (just the last opinion to predict the next one)
        zero_data = copy.copy(true.get_data())
        zero_data.pop()
        zero_data.insert(0, zero_data[0])
        zero = Dataset.create_from_data(zero_data)

        # Calculate the difference between the true and zero datasets
        zero_diff = dataset_difference(true, zero, method="wasserstein")

        # Optimization process and time it
        start = time.time()
        comparison_model = model_class()
        optimizer = optimizers.get_optimizer()
        best_params = optimizer(true, comparison_model, obj_f=optimizers.hyperopt_objective_from_true)
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
        explained_variances.append(basic_var / noisy_var)
        score_diffs.append(zero_diff - opt_mean_diff)

    if not os.path.exists(f"plots/{model_type}/noise"):
        os.makedirs(f"plots/{model_type}/noise")

    # Plot the explained variances
    plt.scatter(explained_variances, score_diffs)
    plt.xlabel("Explained Variance")
    plt.ylabel("Score difference")
    plt.title("Score difference for different explained variance levels")
    plt.savefig(f"plots/{model_type}/noise/score_diff_explained_var_{i}.png")
    plt.close()