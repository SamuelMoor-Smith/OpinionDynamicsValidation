from models.hk_averaging import HKAveragingModel
from experiments.experiment import run_experiment

# for i in range(1, 11):
    # Run the experiment
run_experiment(
    model_class=HKAveragingModel,
    i=6,
    max_noise=0.5
)