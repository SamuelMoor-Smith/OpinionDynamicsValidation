from datasets.dataset import Dataset
from utils.differences import calculate_mean_std
from utils.plotting import plot_2_datasets_snapshots
from utils import optimizers
from models.model import Model
from models.duggins import DugginsModel
import time
from utils.logging import write_results_to_file

def no_noise_experiment(
        model_class: Model
    ):

    model_name = model_class.get_model_name()

    base_model: Model = model_class() # Create model with random parameters
    initial_opinions = base_model.generate_initial_opinions() # generate random initial opinions

    if isinstance(base_model, DugginsModel):
        base_model.sample_isc_for_agents(initial_opinions)

    # Now create 10 datasets with the same model and initial opinions...
    datasets = [Dataset.create_with_model_from_initial(base_model, initial_opinions, num_steps=9) for _ in range(10)]

    # Set the true dataset to be the first of the datasets
    true = datasets[0]

    # Calculate mean and std of differences between the first dataset and the rest
    base_mean_diff, base_std_diff = calculate_mean_std(true, datasets[1:], "Baseline", method="wasserstein")

    # Plot the first and second datasets
    plot_2_datasets_snapshots(true, datasets[1], difference="wasserstein", path=f"plots/{model_name}/no_noise/same/")

    # Optimization process and time it
    start = time.time()

    if isinstance(base_model, DugginsModel):
        comparison_model: Model = DugginsModel(agents=base_model.get_cleaned_agents())
    else:
        comparison_model: Model = model_class()

    optimizer = optimizers.get_optimizer()

    opt_params = {"from_true": False, "num_snapshots": 10}
    best_params = optimizer(true, comparison_model, opt_params, obj_f=optimizers.hyperopt_objective)

    print(f"Optimization took {time.time() - start} seconds")

    # Set the best parameters
    comparison_model.set_normalized_params(best_params)

    # Print both params
    print("Baseline model params: ", base_model.params)
    print("Optimized model params: ", comparison_model.params)

    # Now create 10 more datasets with the optimized model and initial opinions
    opt_datasets = [Dataset.create_with_model_from_initial(comparison_model, initial_opinions, num_steps=9) for _ in range(10)]

    # Calculate mean and std of differences between the first dataset and the rest
    opt_mean_diff, opt_std_diff = calculate_mean_std(true, opt_datasets, "Optimized", method="wasserstein")

    # Plot the true dataset and the first of the optimized
    plot_2_datasets_snapshots(true, opt_datasets[0], difference="wasserstein", path=f"plots/{model_name}/no_noise/diff/")

    # Write the results to a file
    write_results_to_file(
        base_model.params, best_params, 
        base_mean_diff, base_std_diff, 
        opt_mean_diff, opt_std_diff, 
        path=f"results/{model_name}/no_noise/"
    )


