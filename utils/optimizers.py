from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
from datasets.dataset import Dataset
import numpy as np
from utils.differences import dataset_difference, snapshot_difference
from models.model import Model
import time
from models.duggins import DugginsModel
from multiprocessing import Pool

# Set Hyperopt logger to display only errors
logger = logging.getLogger("hyperopt.tpe")
logger.setLevel(logging.ERROR)

def get_optimizer():
    return hyperopt()

def hyperopt():
    """
    Get the optimizer function based on the name.
    """
    best = lambda true, model, opt_params, obj_f=hyperopt_objective: fmin(
        fn=lambda params: obj_f(true, model, params, opt_params),
        space={param: hp.uniform(param, 0, 1) for param in model.params.keys()},
        algo=tpe.suggest,
        max_evals=500,
        trials=Trials(),
        show_progressbar=True
    )
    return best

def hyperopt_objective(true: Dataset, model: Model, model_params, opt_params):
    """Objective function for Hyperopt to minimize"""
    model.set_normalized_params(model_params)
    diffs = []
    for _ in range(5):
        scores = run_and_score_optimal(true, model, opt_params)
        # normalizer = opt_params["zero_diff1to6"] if "zero_diff1to6" in opt_params else 1
        normalizer = 1
        diffs.append(np.sum(scores)/normalizer)
    return {
        'loss': np.mean(diffs),
        'status': STATUS_OK,
    }

def run_and_score_optimal(true: Dataset, model: Model, opt_params):
    """Run and score the model optimally."""
    true_data = true.get_data()
    if isinstance(model, DugginsModel):
        model.sample_isc_for_agents(true_data[0])
    end_index = min(len(true_data) - 1, opt_params["num_snapshots"])
    ops = true_data[0]
    scores = [0]
    for i in range(1,6):
        if opt_params["from_true"]: ops = model.run(true_data[i-1])
        else: ops = model.run(ops)
        scores.append(snapshot_difference(ops, true_data[i], method="wasserstein", range=model.get_opinion_range()))
    return scores