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

# Run this experiment 10 times
for i in range(10):

    # generate random initial opinions
    initial_opinions = create_random_opinion_distribution(N=1000, min_val=0, max_val=1, seed=i)

    # Create a model with random parameters
    base_model = DeffuantModel()
    print("Model created with random parameters: ", base_model.params)

    # Create the array of explained variances and score differences
    explained_variances = []
    score_diffs = []
    
    for noise in np.linspace(0, 0.5, 15):

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
        comparison_model = DeffuantModel()
        optimizer = optimizers.get_optimizer()
        best_params = optimizer(true, comparison_model, obj_f=optimizers.hyperopt_objective_noisy)
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

    # Plot the explained variances
    plt.plot(explained_variances, score_diffs)
    plt.xlabel("Explained Variance")
    plt.ylabel("Score difference")
    plt.title("Score difference for different explained variance levels")
    plt.savefig("plots/deffuant/noise/score_diff_explained_var.png")
    plt.close()