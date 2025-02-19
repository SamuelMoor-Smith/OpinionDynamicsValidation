from models.carpentras import CarpentrasModel
from models.hk_averaging import HKAveragingModel
from models.deffuant import DeffuantModel
from experiments.varying_noise import varying_noise_experiment

for i in range(1, 11):
    # Run the experiment
    varying_noise_experiment(
        model_class=CarpentrasModel,
        i=i
    )