from models.carpentras import CarpentrasModel
from experiments.real_data import real_data_experiment

# Run the experiment
real_data_experiment(
    model_class=CarpentrasModel,
    model_type="carpentras",
    data_header="imwbcnt",
    scale=5,
    adjust=1
)