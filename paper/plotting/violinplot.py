import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datasets.ess.header_info import ess_header_info

# Model name mapping
model_info = {
    "deffuant": ("Deffuant Model", "C0"),
    "hk_averaging": ("HK Averaging Model", "C1"),
    "carpentras": ("ED Model", "C4"),
    "duggins": ("Duggins Model", "C2"),
}

# Create palette based on model_info
palette = {info[0]: info[1] for info in model_info.values()}

dfs = []
for model in model_info.keys():
    df = pd.read_csv(f"paper/real/{model}.csv")
    df["Model Title"] = model_info[model][0]
    dfs.append(df)

# Combine all DataFrames
df_combined = pd.concat(dfs, ignore_index=True)
df_combined["Dataset"] = df_combined["Key"].map(lambda k: f"{k}-{ess_header_info[k]['country'][:2].upper()}")

# Plot

TITLE = f"Model Performance on ESS Data"
Y_LABEL = "Explained Variance"
X_LABEL = "ESS Dataset"

plt.figure(figsize=(12, 6))
sns.stripplot(data=df_combined, x="Dataset", y="Scaled Optimized Difference", hue="Model Title", dodge="quartile", alpha=0.5, palette=palette, size=5)

plt.axhline(y=0, color='black', linewidth=2)

plt.title(TITLE, fontsize=20)

plt.ylabel(Y_LABEL, fontsize=18)
plt.xlabel(X_LABEL, fontsize=18)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig("paper/figures/real/stripplot.png", dpi=300, bbox_inches="tight")
plt.show()
