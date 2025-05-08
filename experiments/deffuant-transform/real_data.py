from models.deffuant_transform import TransformDeffuantModel
from experiments.real_data import real_data_experiment
from datasets.ess.header_info import ess_header_info

for key, value in ess_header_info.items():
    # Run the experiment
    real_data_experiment(
        model_class=TransformDeffuantModel,
        data_header=key
    )