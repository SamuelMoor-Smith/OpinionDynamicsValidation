from models.duggins import DugginsModel
from experiments.experiment import run_experiment

# for i in range(1, 11):
# Run the experiment
run_experiment(
    model_class=DugginsModel,
    i=1,
    max_noise=0
)