import argparse
import copy
import json

from datasets.dataset import Dataset
from datasets.ess.ess_file import ESSFile
from datasets.ess.header_info import ess_header_info
from experiments.common import (build_model, distort_name, get_model_class, run_optimizer)
from models.duggins import DugginsModel
from models.model import Model
from utils.differences import dataset_difference
from utils.paths import res_file, use_path


def get_true_data(key: str):
    """Get the true data for a given ESS key."""
    key_info = ess_header_info[key]
    essfile = ESSFile(
        f'datasets/ess/ess_datasets/{key_info["folder"]}',
        key=key,
        key_info=key_info,
        country=key_info["country"],
        model_range=PredictionModelClass.get_opinion_range()
    )
    return key_info, essfile, essfile.get_true()

def predict_ess_key_data(key: str, seed=0):

    print(f"Running experiment for {prediction_model_name} on ESS key: {key}...")

    key_info, essfile, true_data = get_true_data(key)

    # Create null model (just the last opinion to predict the next one) and calculate the `opinion_drift` of the dataset - the difference between the true and null model datasets
    null_model_data = Dataset.create_null_model_dataset(true_data, None)
    opinion_drift = dataset_difference(true_data, null_model_data)

    for trial in range(KEY_SC):

        trial_info = {
            "trial": trial,
            "ess_key": key,
            "ess_country": key_info["country"],
            "prediction_model": prediction_model_name,
            "opinion_drift": opinion_drift
        }

        if prediction_model_name == "duggins":
            prediction_model: Model = DugginsModel(seed=seed, n=essfile.get_min_agents())
        else:
            prediction_model: Model = build_model(PredictionModelClass, args.distort_prediction, seed, hk_type=prediction_hk_type)

        # Optimization process
        run_optimizer(true_data, prediction_model)

        # For self-consistency, create TRIAL_SC datasets with the `prediction_model` and the `true_data` as the input
        predictions = [Dataset.create_with_model_from_true(prediction_model, true_data.get_data()) for _ in range(TRIAL_SC)]

        trial_info["params"] = str(prediction_model.params)

        for subtrial in range(TRIAL_SC):

            loss = dataset_difference(true_data, predictions[subtrial])

            subtrial_info = copy.deepcopy(trial_info)
            subtrial_info["subtrial"] = subtrial
            subtrial_info["loss"] = loss

            seed += 1

            # Save the subtrial info to a file
            with open(results_file, "a") as f:
                f.write(json.dumps(subtrial_info) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction_model", type=str, default=None)
    parser.add_argument("--distort_prediction", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    seed = args.seed

    # Extract the arguments
    prediction_model_name = args.prediction_model
    PredictionModelClass, prediction_hk_type = get_model_class(prediction_model_name)
    prediction_model_name = distort_name(prediction_model_name, args.distort_prediction)

    KEY_SC = 10
    TRIAL_SC = 10

    # Create the results directory if it doesn't exist
    results_path = use_path(f"results/real")
    results_file = res_file(results_path, None, prediction_model_name)

    # Loop through the ESS keys
    for key in ess_header_info.keys():
        predict_ess_key_data(key, seed)
