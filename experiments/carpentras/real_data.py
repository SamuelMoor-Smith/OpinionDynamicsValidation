from models.carpentras import CarpentrasModel
from experiments.real_data import real_data_experiment

# Run the experiment
real_data_experiment(
    model_class=CarpentrasModel,
    data_header="lrscale",
    scale=0.2,
    adjust=-1
)