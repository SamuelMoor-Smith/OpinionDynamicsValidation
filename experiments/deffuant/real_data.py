from models.deffuant import DeffuantModel
from experiments.real_data import real_data_experiment
from datasets.ess.header_info import ess_header_info

for key, value in ess_header_info.items():
    # Run the experiment
    real_data_experiment(
        model_class=DeffuantModel,
        data_header=key
    )