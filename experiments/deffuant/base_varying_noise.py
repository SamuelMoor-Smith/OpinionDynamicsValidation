from models.deffuant import DeffuantModel
from experiments.old.old_base_code import base_varying_noise_experiment

# for i in range(1, 11):
# Run the experiment
base_varying_noise_experiment(
    model_class=DeffuantModel,
    i=6
)