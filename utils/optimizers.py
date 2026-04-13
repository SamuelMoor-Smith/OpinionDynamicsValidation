from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
from datasets.dataset import Dataset
import numpy as np
import random
from utils.differences import snapshot_difference
from models.model import Model
from models.duggins import DugginsModel
from contextlib import contextmanager

# Set Hyperopt logger to display only errors
logger = logging.getLogger("hyperopt.tpe")
logger.setLevel(logging.ERROR)

MAX_EVALS = 1250
SC_TRIALS = 5
T_OPT = 5


@contextmanager
def seeded_random_state(seed):
    """Temporarily fix Python and NumPy RNG state for reproducible scoring."""
    np_state = np.random.get_state()
    py_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)

def get_optimizer():
    return hyperopt()

def hyperopt():
    """
    Get the optimizer function based on the name.
    """
    best = lambda true, model, obj_f=safe_objective: fmin(
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
    base_seed = 0 if model.seed is None else model.seed
    for trial_idx in range(SC_TRIALS):
        trial_seed = base_seed + trial_idx
        with seeded_random_state(trial_seed):
            scores = run_and_score_optimal(true, model)
        diffs.append(np.sum(scores))
    return {
        'loss': np.mean(diffs),
        'status': STATUS_OK,
    }

def safe_objective(true: Dataset, model: Model, model_params):
    try:
        return hyperopt_objective(true, model, model_params)
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return {'loss': 1e6, 'status': STATUS_OK}

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
