from models.hk_averaging import HKAveragingModel
from experiments.no_noise import no_noise_experiment

# Run the experiment
no_noise_experiment(
    model_class=HKAveragingModel,
    model_type="hk_averaging"
)