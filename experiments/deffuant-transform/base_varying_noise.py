from models.deffuant_transform import TransformDeffuantModel
from experiments.base_performance import base_varying_noise_experiment

# for i in range(1, 11):
# Run the experiment
base_varying_noise_experiment(
    model_class=TransformDeffuantModel,
    i=1
)