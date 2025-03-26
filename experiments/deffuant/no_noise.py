from models.deffuant import DeffuantModel
from experiments.no_noise import no_noise_experiment

for i in range(20):
    # Run the experiment
    no_noise_experiment(
        model_class=DeffuantModel
    )