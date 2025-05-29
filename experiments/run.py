import os
import json
import time
import argparse

from datasets.dataset import Dataset
from utils.differences import calculate_mean_std
from utils import optimizers

from models.model import Model
from models.distortion import DistortionAdaptor, plot_distortion, BetaCDFTransformation
from models.deffuant import DeffuantModel
from models.hk_averaging import HKAveragingModel
from models.carpentras import CarpentrasModel
from models.duggins import DugginsModel
from utils.plotting.plotting import produce_figure

from utils.differences import dataset_difference

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--true_model", type=str, required=True)
    parser.add_argument("--distort_true", action="store_true")

    parser.add_argument("--prediction_model", type=str, default="same_as_true")
    parser.add_argument("--distort_prediction", action="store_true")

    parser.add_argument("--experiment", type=str, default="reproducibility", choices=["reproducibility", "noise", "optimized"])
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Extract the arguments
    true_model_name = args.true_model
    prediction_model_name = true_model_name if args.prediction_model == "same_as_true" else args.prediction_model
    starting_seed = args.seed
    experiment = args.experiment

    # Get the actual model classes
    TrueModelClass = Model.get_registry()[true_model_name]
    PredictionModelClass = Model.get_registry()[prediction_model_name]

    # Add distortion to the names if needed
    if args.distort_true:
        true_model_name = f"distorted_{true_model_name}"
        if args.prediction_model == "same_as_true":
            # Which means we need to distort it as well
            args.distort_prediction = True

    if args.distort_prediction:
        prediction_model_name = f"distorted_{prediction_model_name}"

    TOTAL_TRIALS = 100
    TRIAL_SC = 10
    STEPS = 9
    MAX_NOISE = 0.5

    # Create the results directory if it doesn't exist
    results_path = f"results/{experiment}"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Create the results file name with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prediction_model_name != true_model_name:
        results_file = f"{results_path}/{true_model_name}_{prediction_model_name}_{timestamp}.jsonl"
    else:
        results_file = f"{results_path}/{true_model_name}_{timestamp}.jsonl"

    seed = starting_seed
    for trial in range(TOTAL_TRIALS):

        print(f"Running trial {trial + 1}/{TOTAL_TRIALS} with seed {seed}")

        trial_info = {
            "trial": trial,
            "true_model": true_model_name,
            "prediction_model": prediction_model_name,
            "seed": seed,
            "experiment": experiment
        }

        # Create true model with random parameters
        true_model: Model = TrueModelClass(seed=seed)
        if args.distort_true:
            true_model = DistortionAdaptor(true_model, seed=seed)
            # plot_distortion(BetaCDFTransformation(), a=true_model.params["a"], b=true_model.params["b"], title="Beta CDF Distortion", show_inverse=True)
        seed += 1

        # generate random initial opinions
        initial_opinions = true_model.generate_initial_opinions()

        # Sample ISC for agents if the model is Duggins
        if isinstance(true_model, DugginsModel):
            true_model.sample_isc_for_agents(initial_opinions)

        # Is the experiment with noise?
        noise = MAX_NOISE * (trial / TOTAL_TRIALS) if experiment == "noise" else 0
        trial_info["noise"] = noise

        # Create the true data
        true_data = Dataset.create_with_model_from_initial_opinions(true_model, initial_opinions, num_steps=STEPS, noise=noise)

        # Create zero data (just the last opinion to predict the next one) and
        # Calculate the `opinion_drift` of the dataset - the difference between the true and null model datasets
        null_model_data = Dataset.create_null_model_dataset(true_data, true_model)
        trial_info["opinion_drift"] = dataset_difference(true_data, null_model_data)

        # No matter what we will use test the true model as the prediction model for our baseline data (reproducibility experiment)
        # For self-consistency, create TRIAL_SC datasets with the `true_model` and the `true_data` as the input
        baseline_predictions = [Dataset.create_with_model_from_true(true_model, true_data.get_data()) for _ in range(TRIAL_SC)]
        trial_info["mean_loss_baseline"], trial_info["std_loss_baseline"] = calculate_mean_std(true_data, baseline_predictions)

        if experiment != "reproducibility": # We will use the optimizer for all other experiments

            if isinstance(true_model, DugginsModel):
                prediction_model: Model = DugginsModel(agents=true_model.get_cleaned_agents())
            else:
                prediction_model: Model = PredictionModelClass()
                if args.distort_prediction:
                    prediction_model = DistortionAdaptor(prediction_model, seed=seed)

            # Optimization process and time it
            start = time.time()
            optimizer = optimizers.get_optimizer()
            best_params = optimizer(true_data, prediction_model, obj_f=optimizers.hyperopt_objective)
            print(f"Optimization took {time.time() - start} seconds")

            # Set the best parameters
            prediction_model.set_normalized_params(best_params)
            print("Best parameters: ", prediction_model.params)

            # For self-consistency, create TRIAL_SC datasets with the `prediction_model` and the `true_data` as the input
            optimizer_predictions = [Dataset.create_with_model_from_true(prediction_model, true_data.get_data()) for _ in range(TRIAL_SC)]
            trial_info["mean_loss_optimizer"], trial_info["std_loss_optimizer"] = calculate_mean_std(true_data, optimizer_predictions)

        with open(results_file, "a") as f:
            f.write(json.dumps(trial_info) + "\n")
        
        print(f"Trial {trial + 1}/{TOTAL_TRIALS} completed.\n\n\n")

    # Plot the results
    produce_figure(
        model=prediction_model_name,
        filepath=results_file,
        experiment=experiment
    )

