from models.duggins import DugginsModel
from experiments.real_data import real_data_experiment

# Run the experiment
real_data_experiment(
    model_class=DugginsModel,
    data_header="pplfair"
)