from datasets.ess.ess_file import ESSFile
from datasets.dataset import Dataset
from models.deffuant import DeffuantModel
from models.deffuant_transform import TransformDeffuantModel
from models.hk_averaging import HKAveragingModel
from models.carpentras import CarpentrasModel
from models.duggins import DugginsModel
from utils import optimizers
from models.model import Model
import time
from utils.differences import dataset_difference, calculate_mean_std
from datasets.ess.header_info import ess_header_info
import argparse
import copy
import os
import json

def predict_ess_key_data(key: str):

    print(f"Running experiment for {prediction_model_name} on ESS key: {key}...")

    # Get the ESS file
    key_info = ess_header_info[key]
    essfile = ESSFile(
        f'datasets/ess/full_groups/{key_info["folder"]}',
        key=key,
        key_info=key_info,
        country=key_info["country"],
        model_range=prediction_model_class.get_opinion_range()
    )
    true_data = essfile.get_true()

    # Create zero data (just the last opinion to predict the next one) and
    # Calculate the `opinion_drift` of the dataset - the difference between the true and null model datasets
    null_model_data = Dataset.create_null_model_dataset(true_data, None)
    opinion_drift = dataset_difference(true_data, null_model_data, method="wasserstein")

    for trial in range(KEY_SC):

        trial_info = {
            "trial": trial,
            "ess_key": key,
            "ess_country": key_info["country"],
            "prediction_model": prediction_model_name,
            "opinion_drift": opinion_drift
        }

        if prediction_model_name == "duggins":
            prediction_model: Model = DugginsModel(n=essfile.get_min_agents())
        else:
            prediction_model: Model = prediction_model_class()

        # Optimization process and time it
        start = time.time()
        optimizer = optimizers.get_optimizer()
        best_params = optimizer(true_data, prediction_model, obj_f=optimizers.hyperopt_objective)
        print(f"Optimization took {time.time() - start} seconds")

        # Set the best parameters
        prediction_model.set_normalized_params(best_params)
        print("Best parameters: ", prediction_model.params)

        # For self-consistency, create TRIAL_SC datasets with the `prediction_model` and the `true_data` as the input
        predictions = [Dataset.create_with_model_from_true(prediction_model, true_data.get_data()) for _ in range(TRIAL_SC)]

        trial_info["params"] = str(prediction_model.params)

        for subtrial in range(TRIAL_SC):

            loss = dataset_difference(true_data, predictions[subtrial])

            subtrial_info = copy.deepcopy(trial_info)
            subtrial_info["subtrial"] = subtrial
            subtrial_info["loss"] = loss

            # Save the subtrial info to a file
            with open(results_file, "a") as f:
                f.write(json.dumps(subtrial_info) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_model", type=str, required=True)
    args = parser.parse_args()

    # Extract the arguments
    prediction_model_name = args.prediction_model

    # Get the actual model classes (going to put in Model class)
    models_to_Models = {
        "deffuant": DeffuantModel,
        "hk_averaging": HKAveragingModel,
        "ed": CarpentrasModel,
        "duggins": DugginsModel,
        "transform_deffuant": TransformDeffuantModel
    }
    prediction_model_class = models_to_Models[prediction_model_name]

    KEY_SC = 10
    TRIAL_SC = 10

    # Create the results directory if it doesn't exist
    results_path = f"results/real"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Create a timestamp for the results file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_path}/{prediction_model_name}_{timestamp}.jsonl"

    # Loop through the ESS keys
    for key in ess_header_info.keys():
        predict_ess_key_data(key)