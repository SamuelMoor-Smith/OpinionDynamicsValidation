import time

from models.model import Model
from models.distortion import DistortionAdaptor
from models.deffuant import DeffuantModel
from models.hk_averaging import HKAveragingModel
from models.carpentras import CarpentrasModel
from models.duggins import DugginsModel
from models.gestefeld_lorenz import GestefeldLorenz
from models.deffuant_with_repulsion import DeffuantWithRepulsionModel
from utils import optimizers
from utils.differences import calculate_mean_std
from utils.plotting.plotting import plot_2_datasets_snapshots
from datasets.dataset import Dataset

def create_preds_w_mean_std(model, true_data, trials=9, group="Baseline"):
    """Create predictions using the model and calculate the mean and std of the differences from the true data."""

    start_time = time.time()
    preds = [Dataset.create_with_model_from_true(model, true_data.get_data()) for _ in range(trials)]
    mean, std = calculate_mean_std(true_data, preds)
    print(f"{group} predictions created in {time.time() - start_time} seconds")

    return preds, mean, std

MODEL_REGISTRY = {
    "deffuant": DeffuantModel,
    "hk_averaging": HKAveragingModel,
    "ed": CarpentrasModel,
    "duggins": DugginsModel,
    "gestefeld_lorenz": GestefeldLorenz,
    "deffuant_with_repulsion": DeffuantWithRepulsionModel,
}

def get_model_class(model_name: str):
    """Get the model class from the registry."""
    hk_type = None
    if model_name.startswith("hk_averaging-"):
        model_name, hk_type = model_name.split("-", 1)
    return MODEL_REGISTRY[model_name], hk_type

def distort_name(model_name: str, distort: bool):
    """Add 'distorted_' prefix to the model name if distort is True."""
    return f"distorted_{model_name}" if distort else model_name

def build_model(ModelClass, distort, seed, hk_type=None, agents=None):
    """Build the model, optionally with distortion."""

    if agents:  
        model: Model = DugginsModel(seed=seed, agents=agents)
    elif hk_type is not None:
        model: Model = HKAveragingModel(seed=seed, method=hk_type)
    else:
        model: Model = ModelClass(seed=seed)

    return DistortionAdaptor(model, seed=seed) if distort else model

def run_optimizer(true_data, prediction_model):
    """Run the optimizer to find the best parameters for the prediction model."""

    # Optimization process and time it
    start = time.time()
    optimizer = optimizers.get_optimizer()
    best_params = optimizer(true_data, prediction_model, obj_f=optimizers.safe_objective)
    print(f"Optimization took {time.time() - start} seconds")

    # Set the best parameters
    prediction_model.set_normalized_params(best_params)
    print("Best parameters: ", prediction_model.params)

def save_high_drift(trial_info, true_model, true_data, null_model_data):
    """Save high drift cases for further analysis."""
    if trial_info["opinion_drift"] > 0.5:
        print(true_model.params)
        plot_2_datasets_snapshots(
            true_data,
            null_model_data,
            path="./results/high_drift"
        )