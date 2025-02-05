from models.hk_averaging import HKAveragingModel
from experiments.real_data import real_data_experiment

# Run the experiment
real_data_experiment(
    model_class=HKAveragingModel,
    model_type="hk_averaging",
    data_header="imwbcnt",
)