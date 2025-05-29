from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
from datasets.dataset import Dataset
import numpy as np
from utils.differences import snapshot_difference
from models.model import Model
from models.duggins import DugginsModel

# Set Hyperopt logger to display only errors
logger = logging.getLogger("hyperopt.tpe")
logger.setLevel(logging.ERROR)

MAX_EVALS = 250
T_OPT = 5

def get_optimizer():
    return hyperopt()

def hyperopt():
    """
    Get the optimizer function based on the name.
    """
    best = lambda true, model, obj_f=hyperopt_objective: fmin(
        fn=lambda params: obj_f(true, model, params),
        space={param: hp.uniform(param, 0, 1) for param in model.params.keys()},
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=Trials(),
        show_progressbar=True
    )
    return best

def hyperopt_objective(true: Dataset, model: Model, model_params):
    """Objective function for Hyperopt to minimize"""
    model.set_normalized_params(model_params)
    diffs = []
    for _ in range(T_OPT):
        scores = run_and_score_optimal(true, model)
        diffs.append(np.sum(scores))
    return {
        'loss': np.mean(diffs),
        'status': STATUS_OK,
    }

def run_and_score_optimal(true: Dataset, model: Model):
    """Run and score the model optimally."""
    true_data = true.get_data()
    if isinstance(model, DugginsModel):
        model.sample_isc_for_agents(true_data[0])
    opinions = true_data[0]
    scores = [0]
    for i in range(1,T_OPT):
        opinions = model.run(true_data[i-1])
        scores.append(snapshot_difference(opinions, true_data[i], range=model.get_opinion_range()))
    return scores