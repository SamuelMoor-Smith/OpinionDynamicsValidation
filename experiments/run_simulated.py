import argparse
import json

from datasets.dataset import Dataset
from experiments.common import (build_model, create_preds_w_mean_std, distort_name, get_model_class, run_optimizer, save_high_drift)
from models.duggins import DugginsModel
from models.model import Model
from utils.differences import dataset_difference
from utils.noise import get_noise
from utils.paths import res_file, use_path
from utils.plotting.plotting import (plot_2_datasets_snapshots, plot_dataset_snapshots, produce_figure)
from utils.rand_gen import increment_seed

SAVE_HIGH_DRIFT = False
TOTAL_TRIALS = 100
TRIAL_SC = 10
STEPS = 9

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--true_model", type=str, required=True)
    parser.add_argument("--distort_true", action="store_true")

    parser.add_argument("--prediction_model", type=str, default="same_as_true")
    parser.add_argument("--distort_prediction", action="store_true")

    parser.add_argument("--experiment", type=str, default="plot_true", choices=["plot_true", "reproducibility", "noise", "optimized"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--plot_datasets", action="store_true", help="Plot the datasets after each trial")

    args = parser.parse_args()

    # Extract the arguments
    true_model_name = args.true_model
    if args.prediction_model == "same_as_true":
        prediction_model_name = true_model_name
        args.distort_prediction = args.distort_true
    else:
        prediction_model_name = args.prediction_model
    seed = args.seed
    experiment = args.experiment

    # Get the actual model classes
    TrueModelClass, true_hk_type = get_model_class(true_model_name)
    PredictionModelClass, prediction_hk_type = get_model_class(prediction_model_name)

    # Add distortion to the names if needed
    true_model_name = distort_name(true_model_name, args.distort_true)
    prediction_model_name = distort_name(prediction_model_name, args.distort_prediction)

    # Create the results directory if it doesn't exist
    results_path = use_path(f"results/{experiment}/{true_model_name}")
    results_file = res_file(results_path, true_model_name, prediction_model_name)

    for trial in range(TOTAL_TRIALS):

        seed = increment_seed(seed, trial)
        print(f"Running trial {trial + 1}/{TOTAL_TRIALS} with seed {seed}")

        trial_info = {
            "trial": trial,
            "true_model": true_model_name,
            "prediction_model": prediction_model_name,
            "seed": seed,
            "experiment": experiment
        }

        # Create true model with random parameters
        true_model: Model = build_model(TrueModelClass, args.distort_true, seed, hk_type=true_hk_type)

        # generate random initial opinions
        initial_opinions = true_model.generate_initial_opinions(seed=seed)
        # Sample ISC for agents if the model is Duggins
        if isinstance(true_model, DugginsModel):
            true_model.sample_isc_for_agents(initial_opinions)

        # Is the experiment with noise?
        noise = get_noise(trial, TOTAL_TRIALS, max_noise=0.5) if experiment == "noise" else 0
        trial_info["noise"] = noise

        # Create the true data
        true_data = Dataset.create_with_model_from_initial_opinions(true_model, initial_opinions, num_steps=STEPS, noise=noise)

        if args.experiment == "plot_true":
            plot_dataset_snapshots(
                true_data,
                path="./results/tmp",
                bins=100
            )
            break

        # Create zero data (just the last opinion to predict the next one) and
        # Calculate the `opinion_drift` of the dataset - the difference between the true and null model datasets
        null_model_data = Dataset.create_null_model_dataset(true_data, true_model)
        trial_info["opinion_drift"] = dataset_difference(true_data, null_model_data)
        if SAVE_HIGH_DRIFT:
            save_high_drift(trial_info, true_model, true_data, null_model_data)

        # No matter what we will use test the true model as the prediction model for our baseline data (reproducibility experiment)
        # For self-consistency, create TRIAL_SC datasets with the `true_model` and the `true_data` as the input
        baseline_predictions, trial_info["mean_loss_baseline"], trial_info["std_loss_baseline"] = \
            create_preds_w_mean_std(true_model, true_data, trials=TRIAL_SC, group="Baseline")

        if args.plot_datasets:
            plot_2_datasets_snapshots(
                true_data,
                baseline_predictions[0],
                path="./results/tmp"
            )

        if experiment != "reproducibility": # We will use the optimizer for all other experiments
            
            agents = None
            if isinstance(true_model, DugginsModel):
                agents = true_model.get_cleaned_agents()
            prediction_model: Model = build_model(PredictionModelClass, args.distort_prediction, seed, hk_type=prediction_hk_type, agents=agents)

            # Optimization process
            run_optimizer(true_data, prediction_model)

            # For self-consistency, create TRIAL_SC datasets with the `prediction_model` and the `true_data` as the input
            optimizer_predictions, trial_info["mean_loss_optimizer"], trial_info["std_loss_optimizer"] = \
                create_preds_w_mean_std(prediction_model, true_data, trials=TRIAL_SC, group="Optimizer")

        with open(results_file, "a") as f:
            f.write(json.dumps(trial_info) + "\n")
        
        print(f"Trial {trial + 1}/{TOTAL_TRIALS} completed.\n\n\n")

    if args.experiment != "plot_true":
        # Plot the results
        produce_figure(
            generator=true_model_name,
            predictor=prediction_model_name,
            filepath=results_file,
            experiment=experiment
        )
