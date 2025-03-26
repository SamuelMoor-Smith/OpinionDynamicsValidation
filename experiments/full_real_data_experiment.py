import pandas as pd
from experiments.real_data import real_data_experiment
from datasets.ess.header_info import ess_header_info
from models.duggins import DugginsModel
from models.hk_averaging import HKAveragingModel
from models.deffuant import DeffuantModel
from models.carpentras import CarpentrasModel
from models.transform_deffuant import TransformDeffuantModel

# Define output CSV file
OUTPUT_CSV = "results/master_results_mar24-duggins.csv"

# List of models to run
models = [DugginsModel]
# models = [DeffuantModel, HKAveragingModel, CarpentrasModel]

# Initialize an empty list to store results
results_list = []

# Loop over each model and dataset key
for model_class in models:
    for key in ess_header_info.keys():
        print(f"Running experiment for {model_class.__name__} on {key}...")

        # Call real_data_experiment (assuming it returns necessary values)
        model_name, zero_diff, optimized_diff, optimized_params = real_data_experiment(
            model_class=model_class,
            data_header=key
        )

        # Compute additional metrics
        is_optimized_better = optimized_diff < zero_diff

        # Store results in a dictionary
        results_list.append({
            "Model": model_name,
            "Key": key,
            "Zero Difference": zero_diff,
            "Scaled Optimized Difference": (optimized_diff - zero_diff)/zero_diff,
            "Raw Optimized Difference": optimized_diff,
            "Optimized Better": is_optimized_better,
            # "Difference": difference,
            "Optimized Params": optimized_params
        })

# Convert list to DataFrame
df_results = pd.DataFrame(results_list)

# Save to CSV
df_results.to_csv(OUTPUT_CSV, index=False)
