from models.model import Model
import pandas as pd

model_info = Model.get_model_plotting_info()

correlations = {}
for model in model_info.keys():
    df = pd.read_json(f"results/real/{model}.jsonl", lines=True)
    for i in range(len(df)):
        key = df.at[i, "ess_key"]
        if f"{model}-{key}" not in correlations:
            correlations[f"{model}-{key}"] = []
        correlations_i = [df.at[i, "correlations"][j][0] for j in range(1, len(df.at[i, "correlations"]))]
        correlations[f"{model}-{key}"].extend(correlations_i)

print("Correlations:")
for model_key, corr_values in correlations.items():
    print(f"{model_key} correlations: mean: {sum(corr_values) / len(corr_values):.4f}, std: {pd.Series(corr_values).std():.4f}")
    # print(f"  {model_key} correlations: {corr_values[:5]}...")  