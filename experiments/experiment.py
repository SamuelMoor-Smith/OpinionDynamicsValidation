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
import json

def run_experiment(
        model_class: Model,
        i="",
        max_noise=0
    ):

    model_name = model_class.get_model_name()

    # Create non-uniformly sampled noise levels
    noises = np.linspace(0, max_noise, 10)  # Uniformly spaced values between 0 and 1
    # noises = 0.5 * uniform_samples**2  # Square root transformation to bias towards smaller values
    
    print(f"Noises: {noises}")

    for noise in noises:

        new_point = {}
        new_point["noise"] = noise
        new_point["model"] = model_name
        new_point["i"] = i

        base_model: Model = model_class() # Create model with random parameters
        initial_opinions = base_model.generate_initial_opinions() # generate random initial opinions

        if isinstance(base_model, DugginsModel):
            base_model.sample_isc_for_agents(initial_opinions)

        # Iterate over the noise levels and create the true data with that noise value
        true, explained_var = Dataset.create_with_model_from_initial_with_noise(base_model, initial_opinions, num_steps=9, noise=noise)

        # Create zero data (just the last opinion to predict the next one)
        zero = Dataset.create_zero_data_from_true(true, base_model)

        # Calculate the difference between the true and zero datasets
        zero_diff = dataset_difference(true, zero, method="wasserstein")
        new_point["zero_diff"] = zero_diff

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
        new_point["opt_mean_diff"] = opt_mean_diff
        new_point["opt_std_diff"] = opt_std_diff

        # Print and store the score difference between the zero and optimized datasets
        print(f"Score difference for noise {noise}: {zero_diff - opt_mean_diff}")
        new_point["explained_var"] = explained_var
        new_point["score_diff"] = (zero_diff - opt_mean_diff)/zero_diff

        plot_2_datasets_snapshots(true, opt_datasets[0], difference="wasserstein", path=f"paper/figures/frequency/")

        if not os.path.exists(f"results/{model_name}/noise"):
            os.makedirs(f"results/{model_name}/noise")

        results_path = f"results/{model_name}/noise/max_noise_{max_noise}_results_{i}.jsonl"  # JSON Lines format

        with open(results_path, "a") as f:
            f.write(json.dumps(new_point) + "\n")