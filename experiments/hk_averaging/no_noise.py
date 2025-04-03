from models.hk_averaging import HKAveragingModel
from experiments.experiment import run_experiment

# Run the experiment
run_experiment(
    model_class=HKAveragingModel,
    i=1
)