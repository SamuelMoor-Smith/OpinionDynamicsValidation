from models.deffuant_transform import TransformDeffuantModel
from experiments.experiment_transfer import run_experiment_transfer

# for i in range(20):
# Run the experiment
run_experiment_transfer(
    model_class=TransformDeffuantModel,
    i=6
)