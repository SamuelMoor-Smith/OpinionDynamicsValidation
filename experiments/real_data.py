from datasets.ess.ess_file import ESSFile
import numpy as np
from utils.rand_gen import create_random_opinion_distribution
from datasets.dataset import Dataset
from models.duggins import DugginsModel
from utils.differences import calculate_mean_std
from utils.plotting import plot_2_datasets_snapshots
from utils import optimizers
from models.model import Model
import time
from utils.my_logging import write_results_to_file
from utils.differences import dataset_difference
from datasets.ess.header_info import ess_header_info

# _ESSFILE = ESSFile('datasets/ess/combined-sept26.csv', 'imbgeco')
# _ESSFILE.plot_data()

'''
Basically on the first 5, use 1 to predict 2, and evaluate then use 2 to predict 3, then evaluate… all using the same parameters till 5 to predict 6… then evaluate on the next set of them with the same parameters to see how well it preformed - compare this all with the 0 model

For the duggins, I basically have to show that using the same even with the same parameters, the outcomes are vastly different - it is a very uncertain model

compare with zero model similar to in varying noise
'''

def real_data_experiment(
        model_class: Model,
        data_header: str,
    ):

    model_name = model_class.get_model_name()
    print(f"Running {model_name} on {data_header}")

    # Get the ESS file
    key_info = ess_header_info[data_header]
    essfile = ESSFile(
        f'datasets/ess/full_groups/{key_info["folder"]}',
        key=data_header,
        key_info=key_info,
        country=key_info["country"],
        model_range=model_class.get_opinion_range()
    )
    true = essfile.get_true()

    # Get zero data
    zero = Dataset.create_zero_data_from_true(true, None)
    zero_diff = dataset_difference(true, zero, method="wasserstein")
    print(f"Zero difference: {zero_diff}")

    all_diffs = []

    for i in range(10):

        # Optimization process and time it
        start = time.time()
        if model_name == "duggins":
            comparison_model: Model = DugginsModel(n=essfile.get_min_agents())
        else: 
            comparison_model: Model = model_class()
        optimizer = optimizers.get_optimizer()
        opt_params = {"from_true": True, "num_snapshots": 5}
        best_params = optimizer(true, comparison_model, opt_params, obj_f=optimizers.hyperopt_objective)
        print(f"Optimization took {time.time() - start} seconds")

        # Set the best parameters
        comparison_model.set_normalized_params(best_params)

        # Now create 10 more datasets with the optimized model and true data 
        opt_datasets = [Dataset.create_with_model_from_true(comparison_model, true_data=true.get_data()) for _ in range(10)]

        # Calculate the differences and add them to samples
        all_diffs.extend([dataset_difference(true, d) for d in opt_datasets])

    return model_name, zero_diff, all_diffs, comparison_model.params


