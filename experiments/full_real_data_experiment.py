import pandas as pd
from experiments.real_data import real_data_experiment
from datasets.ess.header_info import ess_header_info
from models.deffuant import DeffuantModel
from models.hk_averaging import HKAveragingModel
from models.carpentras import CarpentrasModel
from models.duggins import DugginsModel

# Output path
OUTPUT_CSV = "results/violin_data_apr3-hk_averaging.csv"

# Models to run
models = [HKAveragingModel]

# List to collect all run-level diffs for violin plot
violin_data = []

# Loop through each model and ESS key
for model_class in models:
    for key in ess_header_info.keys():
        print(f"Running experiment for {model_class.__name__} on {key}...")

        model_name, zero_diff, optimized_diffs, optimized_params = real_data_experiment(
            model_class=model_class,
            data_header=key
        )

        for i, (diff, params) in enumerate(zip(optimized_diffs, optimized_params)):
            violin_data.append({
                "Model": model_name,
                "Key": key,
                "Run": i,
                "Zero Difference": zero_diff,
                "Raw Optimized Difference": diff,
                "Scaled Optimized Difference": (zero_diff - diff) / zero_diff,
                "Optimized Better": diff < zero_diff,
                "Optimized Params": str(params)  # store each run's specific params
            })

# Convert to DataFrame
df_violin = pd.DataFrame(violin_data)

# Save
df_violin.to_csv(OUTPUT_CSV, index=False)

