from datasets.ess.ess_file import ESSFile
from datasets.dataset import Dataset
from models.distortion import DistortionAdaptor
from models.duggins import DugginsModel
from utils import optimizers
from models.model import Model
import time
from utils.differences import dataset_difference
from datasets.ess.header_info import ess_header_info
import argparse
import copy
import os
import json
from scipy import stats
from models.model import Model
from models.distortion import DistortionAdaptor, plot_distortion, BetaCDFTransformation
from models.deffuant import DeffuantModel
from models.hk_averaging import HKAveragingModel
from models.carpentras import CarpentrasModel
from models.duggins import DugginsModel
from models.gestefeld_lorenz import GestefeldLorenz
from models.deffuant_with_repulsion import DeffuantWithRepulsionModel

def predict_ess_key_data(key: str):

    print(f"Running experiment for {prediction_model_name} on ESS key: {key}...")

    # Get the ESS file
    key_info = ess_header_info[key]
    essfile = ESSFile(
        f'datasets/ess/ess_datasets/{key_info["folder"]}',
        key=key,
        key_info=key_info,
        country=key_info["country"],
        model_range=PredictionModelClass.get_opinion_range()
    )
    true_data = essfile.get_true()

    # Create zero data (just the last opinion to predict the next one) and
    # Calculate the `opinion_drift` of the dataset - the difference between the true and null model datasets
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
            prediction_model: Model = DugginsModel(n=essfile.get_min_agents())
        else:
            prediction_model: Model = PredictionModelClass()

        # Optimization process and time it
        start = time.time()
        optimizer = optimizers.get_optimizer()
        best_params = optimizer(true_data, prediction_model, obj_f=optimizers.safe_objective)
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
            sorted_true_steps = [sorted(step) for step in true_data.get_data()]
            sorted_predictions_steps = [sorted(step) for step in predictions[subtrial].get_data()]
            subtrial_info["correlations"] = [stats.pearsonr(sorted_true_steps[i], sorted_predictions_steps[i]) for i in range(len(true_data.get_data()))]

            # Save the subtrial info to a file
            with open(results_file, "a") as f:
                f.write(json.dumps(subtrial_info) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction_model", type=str, default=None)
    parser.add_argument("--distort_prediction", action="store_true")

    args = parser.parse_args()

    # Extract the arguments
    prediction_model_name = args.prediction_model
    PredictionModelClass = Model.get_registry()[prediction_model_name]
    if args.distort_prediction:
        prediction_model_name = f"distorted_{prediction_model_name}"

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