import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datasets.ess.header_info import ess_header_info
from models.model import Model

print(sns.plotting_context())

model_info = Model.get_model_plotting_info()

# Create palette based on model_info
palette = {info[0]: info[1] for info in model_info.values()}

dfs = []
for model in model_info.keys():
    if model == "gestefeld_lorenz" or model == "deffuant_with_repulsion":
        # Special case
        df = pd.read_json(f"old/paper/real/{model}.jsonl", lines=True)
    else:
        df = pd.read_csv(f"old/paper/real/{model}.csv")
        df["ess_key"] = df["Key"]

    df["Model Title"] = model_info[model][0]

    if "Scaled Optimized Difference" not in df.columns:
        ev_col = "Scaled Optimized Difference"
        df[ev_col] = 1 - df[f"loss"] / df["opinion_drift"]
        df[ev_col] = df[ev_col].replace([np.inf, -np.inf], np.nan).fillna(df[ev_col].min())

    dfs.append(df)

# Combine all DataFrames
df_combined = pd.concat(dfs, ignore_index=True)
df_combined["Dataset"] = df_combined["ess_key"].map(lambda k: f"{k}-{ess_header_info[k]['country'][:2].upper()}")
df_combined["explained_variance"] = df_combined["Scaled Optimized Difference"]

# Plot

TITLE = f"Model Performance on ESS Data"
Y_LABEL = "Explained Variance"
X_LABEL = "ESS Dataset"

plt.figure(figsize=(12, 6))
sns.stripplot(data=df_combined, x="Dataset", y="explained_variance", hue="Model Title", dodge="quartile", alpha=0.5, palette=palette, size=5)

plt.axhline(y=0, color='black', linewidth=2)

plt.title(TITLE, fontsize=20)

plt.ylabel(Y_LABEL, fontsize=18)
plt.xlabel(X_LABEL, fontsize=18)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig("results/figures/stripplot.png", dpi=300, bbox_inches="tight")
plt.show()

# # Model name mapping
# model_info = {
#     "deffuant": ("Deffuant Model", "C0"),
#     "hk_averaging": ("HK Averaging Model", "C1"),
#     "carpentras": ("ED Model", "C4"),
#     "duggins": ("Duggins Model", "C2"),
#     "deffuant_with_repulsion": ("Deffuant With Repulsion", "C3"),
#     "gestefeld_lorenz": ("Gestefeld-Lorenz Model", "C5"),
# }

# # Create palette based on model_info
# palette = {info[0]: info[1] for info in model_info.values()}

# dfs = []
# for model in model_info.keys():
#     if model == "gestefeld_lorenz" or model == "deffuant_with_repulsion":
#         # Special case
#         df = pd.read_json(f"old/paper/real/{model}.jsonl", lines=True)
#     else:
#         df = pd.read_csv(f"old/paper/real/{model}.csv")
#         df["ess_key"] = df["Key"]

#     df["Model Title"] = model_info[model][0]

#     if "Scaled Optimized Difference" not in df.columns:
#         ev_col = "Scaled Optimized Difference"
#         df[ev_col] = 1 - df[f"loss"] / df["opinion_drift"]
#         df[ev_col] = df[ev_col].replace([np.inf, -np.inf], np.nan).fillna(df[ev_col].min())

#     dfs.append(df)

# # Combine all DataFrames
# df_combined = pd.concat(dfs, ignore_index=True)
# df_combined["Dataset"] = df_combined["ess_key"].map(lambda k: f"{k}-{ess_header_info[k]['country'][:2].upper()}")

# # Plot

# TITLE = f"Model Performance on ESS Data"
# Y_LABEL = "Explained Variance"
# X_LABEL = "ESS Dataset"

# plt.figure(figsize=(12, 6))
# sns.stripplot(
#     data=df_combined, 
#     x="Dataset", 
#     y="Scaled Optimized Difference", 
#     hue="Model Title", 
#     dodge="quartile", 
#     alpha=0.35, 
#     palette=palette, 
#     size=5)
# # 2. Overlay Pointplot: mean and error bars (std)
# sns.pointplot(
#     data=df_combined,
#     x="Dataset",
#     y="Scaled Optimized Difference",
#     hue="Model Title",
#     dodge=0.7,  # Align with stripplot
#     join=False,
#     markers="D",
#     scale=1.0,
#     ci="sd",  # standard deviation; use "se" for standard error, or None for no bars
#     # palette=palette,
#     color='black',  # Use a single color for the pointplot
#     errwidth=2,
#     zorder=10,
#     legend=False
# )

# plt.axhline(y=0, color='black', linewidth=2)

# plt.title(TITLE, fontsize=20)

# plt.ylabel(Y_LABEL, fontsize=18)
# plt.xlabel(X_LABEL, fontsize=18)
# # plt.ylim(-0.2, 0.1)
# plt.xticks(rotation=45, fontsize=16)
# plt.yticks(fontsize=16)

# plt.tight_layout()
# plt.savefig("old/paper/figures/real/stripplot.png", dpi=300, bbox_inches="tight")
# plt.show()
