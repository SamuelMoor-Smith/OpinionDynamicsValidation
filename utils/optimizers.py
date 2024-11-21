from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
from datasets.dataset import Dataset
import numpy as np
from utils.differences import dataset_difference
from models.model import Model

# Set Hyperopt logger to display only errors
logger = logging.getLogger("hyperopt.tpe")
logger.setLevel(logging.ERROR)

def get_optimizer():
    return hyperopt()

def hyperopt():
    """
    Get the optimizer function based on the name.
    """
    return lambda true, model, obj_f=hyperopt_objective: fmin(
        fn=lambda params: obj_f(true, model, params),
        space={param: hp.uniform(param, 0, 1) for param in model.params.keys()},
        algo=tpe.suggest,
        max_evals=300,
        trials=Trials(),
        show_progressbar=True
    )

def hyperopt_objective(true: Dataset, model: Model, params):
    """Objective function for Hyperopt to minimize"""
    model.set_normalized_params(params)
    datasets = [Dataset.create_with_model_from_initial(model, true.get_data()[0], num_steps=9) for _ in range(10)]
    diffs = [dataset_difference(true, d, method="wasserstein") for d in datasets]
    return {
        'loss': np.mean(diffs),
        'status': STATUS_OK,
    }

def hyperopt_objective_noisy(true: Dataset, model: Model, params):
    """Objective function for Hyperopt to minimize"""
    model.set_normalized_params(params)
    datasets = [Dataset.create_with_model_from_true(model, true.get_data()) for _ in range(10)]
    diffs = [dataset_difference(true, d, method="wasserstein") for d in datasets]
    return {
        'loss': np.mean(diffs),
        'status': STATUS_OK,
    }