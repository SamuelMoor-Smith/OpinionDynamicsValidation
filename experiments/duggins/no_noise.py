from models.duggins import DugginsModel
from experiments.no_noise import no_noise_experiment

# Run the experiment
no_noise_experiment(
    model_class=DugginsModel
)
