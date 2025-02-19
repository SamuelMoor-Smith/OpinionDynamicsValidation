from models.deffuant import DeffuantModel
from experiments.varying_noise import varying_noise_experiment

for i in range(1, 11):
    # Run the experiment
    varying_noise_experiment(
        model_class=DeffuantModel,
        i=i
    )