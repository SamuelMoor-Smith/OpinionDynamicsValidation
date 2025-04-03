from models.carpentras import CarpentrasModel
from models.hk_averaging import HKAveragingModel
from models.deffuant import DeffuantModel
from experiments.experiment import run_experiment

# for i in range(1, 11):
# Run the experiment
run_experiment(
    model_class=CarpentrasModel,
    i=6,
    max_noise=0
)