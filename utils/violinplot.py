import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datasets.ess.header_info import ess_header_info

# Load the real results
df_def = pd.read_csv("results/violin_data_apr3-deffuant.csv")
df_hk = pd.read_csv("results/violin_data_apr3-hk_averaging.csv")

# Add model labels to each
df_def["ModelTitle"] = "Deffuant"
# df_hk["ModelTitle"] = "HK Averaging"

# Combine into one DataFrame
df = df_def[df_def["Key"].isin(["stfdem"])].copy()

# pd.concat(df_def, ignore_index=True)

print(df.shape)

# Optional: Format the dataset key (e.g., add country code suffix manually or via map)
# If you have country info:
# from datasets.ess.header_info import ess_header_info
df["Dataset"] = df["Key"].map(lambda k: f"{k}-{ess_header_info[k]['country'][:2].upper()}")
# Else just use the key directly
# df["Dataset"] = df["Key"]

# Plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x="Dataset", y="Scaled Optimized Difference", hue="ModelTitle", inner="quartile", split=False)

plt.title('Model Performance Comparison per ESS Key')
plt.xticks(rotation=45)
plt.ylabel("Scaled Improvement over Zero Model")
plt.xlabel("ESS Dataset")
plt.tight_layout()
plt.savefig("model_comparison_violin_noisy_split.png", dpi=300, bbox_inches="tight")
# plt.show()
