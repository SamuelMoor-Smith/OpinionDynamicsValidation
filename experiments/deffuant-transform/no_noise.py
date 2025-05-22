from models.deffuant_transform import TransformDeffuantModel
from experiments.experiment import run_experiment

# for i in range(20):
# Run the experiment
run_experiment(
    model_class=TransformDeffuantModel,
    i=5
)