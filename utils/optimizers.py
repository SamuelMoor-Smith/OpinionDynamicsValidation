from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
from datasets.dataset import Dataset
import numpy as np
from utils.differences import dataset_difference, snapshot_difference
from models.model import Model
import time
from models.duggins import DugginsModel

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
    diffs = run_and_score_optimal(true, model)
    return {
        'loss': np.mean(diffs),
        'status': STATUS_OK,
    }

def hyperopt_objective_from_true(true: Dataset, model: Model, params):
    """Objective function for Hyperopt to minimize"""
    model.set_normalized_params(params)
    diffs = run_and_score_optimal_from_true(true, model)
    return {
        'loss': np.mean(diffs),
        'status': STATUS_OK,
    }

def run_and_score_optimal(true, model):
    """Run and score the model optimally."""
    if type(model) == DugginsModel:
        model.create_agents(true.get_data()[0])
    true_data = true.get_data()
    ops = true_data[0]
    scores = [0]
    for i in range(1,9):
        ops = model.run(ops)
        scores.append(snapshot_difference(ops, true_data[i], method="wasserstein"))
    return scores

def run_and_score_optimal_from_true(true, model):
    """Run and score the model optimally with noise."""
    true_data = true.get_data()
    scores = [0]
    for i in range(1,9):
        ops = model.run(true_data[i-1])
        scores.append(snapshot_difference(ops, true.get_data()[i], method="wasserstein"))
    return scores