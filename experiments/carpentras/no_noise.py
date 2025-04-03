from models.carpentras import CarpentrasModel
from experiments.experiment import run_experiment

# Run the experiment
run_experiment(
    model_class=CarpentrasModel,
    i=1
)