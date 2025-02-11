from models.deffuant import DeffuantModel
from experiments.real_data import real_data_experiment

# Run the experiment
real_data_experiment(
    model_class=DeffuantModel,
    model_name="deffuant",
    data_header="imwbcnt",
)