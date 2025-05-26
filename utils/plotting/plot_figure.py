import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from plotting.plotting_utils import get_yx_fit_y_lower_upper

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="reproducibility")
# parser.add_argument("--path1", type=str, required=True)
# parser.add_argument("--path2", type=str, default=None)
# parser.add_argument("--path3", type=str, default=None)
# parser.add_argument("--path4", type=str, default=None)
args = parser.parse_args()

# Get x parameter
if args.experiment == "noise":
    x_param = "noise"
else:
    x_param = "zero_diff"

# Model name mapping
model_info = {
    "deffuant": ("Deffuant Model", "C0"),
    "hk_averaging": ("HK Averaging Model", "C1"),
    "carpentras": ("ED Model", "C4"),
    "duggins": ("Duggins Model", "C2"),
    "transform_deffuant": ("Transformed Deffuant Model", "C3"),
}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, model in enumerate(["transform_deffuant", "deffuant", "hk_averaging", "carpentras"]): 

    filepath = f"./paper/{args.experiment}/{model}.jsonl"
    if filetype == "csv":
        df = pd.read_csv(filepath)
    elif filetype == "jsonl":
        with open(filepath, "r") as f:
            json_data = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(json_data)

    if args.experiment == "optimized":
        filepath_base = f"./paper/reproducibility/{model}"
        if model == "transform_deffuant":
            filetype_base = "jsonl"
            with open(f"{filepath_base}.{filetype_base}", "r") as f:
                json_data = [json.loads(line) for line in f if line.strip()]
            df_base = pd.DataFrame(json_data)
        else:
            filetype_base = "csv"
            df_base = pd.read_csv(f"{filepath_base}.{filetype_base}")
        df_base["explained_variance"] = 1 - df_base["opt_mean_diff"]/df_base["zero_diff"]
        df_base["explained_variance"] = df_base["explained_variance"].replace([np.inf, -np.inf], np.nan).fillna(df_base["explained_variance"].min())

    # Compute explained variance
    df["explained_variance"] = 1 - df["opt_mean_diff"]/df["zero_diff"]
    df["explained_variance"] = df["explained_variance"].replace([np.inf, -np.inf], np.nan).fillna(df["explained_variance"].min())

    # Get the fit
    xfit, yfit, ylower, yupper = get_yx_fit_y_lower_upper(df, x_param)
    if args.experiment == "optimized":
        xfit_base, yfit_base, ylower_base, yupper_base = get_yx_fit_y_lower_upper(df_base, x_param)

    # # Plot
    # plt.figure(figsize=(8, 6))

    COLOR = model_info[model][1]
    DATA_TYPE = "Reproduced" if args.experiment == "reproducibility" else "Optimized"
    FIT_LABEL = f"{DATA_TYPE} Exponential Fit" if args.experiment == "noise" else f"{DATA_TYPE} Logarithmic Fit"
    RAW_LABEL = f"{DATA_TYPE} Raw Data" if args.experiment == "noise" else f"{DATA_TYPE} Raw Data"
    
    # plt.scatter(df[x_param], df["explained_variance"], alpha=0.2, label=RAW_LABEL, color=COLOR)
    # plt.plot(xfit, yfit, label=FIT_LABEL, color=COLOR)
    # plt.fill_between(xfit, ylower, yupper, alpha=0.2, color=COLOR)

    ax = axs[i]

    ax.scatter(df[x_param], df["explained_variance"], alpha=0.2, label=RAW_LABEL, color=COLOR)
    ax.plot(xfit, yfit, label=FIT_LABEL, color=COLOR)
    ax.fill_between(xfit, ylower, yupper, alpha=0.2, color=COLOR)

    if args.experiment == "optimized":
        ax.scatter(df_base[x_param], df_base["explained_variance"], alpha=0.2, label="Reproduced Raw Data", color="#808080")
        ax.plot(xfit_base, yfit_base, label="Reproduced Logarithmic Fit", color="#808080")
        ax.fill_between(xfit_base, ylower_base, yupper_base, alpha=0.2, color="#808080")

    # Labels and legend
    TITLE = f"{model_info[model][0]} {args.experiment.capitalize()}"
    if args.experiment == "noise":
        TITLE = f"{model_info[model][0]} with Noise"
    Y_LABEL = "Explained Variance"
    X_LABEL = "Noise" if args.experiment == "noise" else "Opinion Drift"


    ax.set_title(TITLE, fontsize=24)
    # plt.xlabel(X_LABEL, fontsize=16)
    # plt.ylabel(Y_LABEL, fontsize=16)

    # Axis labels only on left/bottom
    if i % 2 == 0:
        ax.set_ylabel(Y_LABEL, fontsize=22)
    if i % 2 != 0:
        ax.set_yticklabels([])
    if i >= 2:
        ax.set_xlabel(X_LABEL, fontsize=22)
    # if i < 2:
    # ax.set_xticklabels([])
    ax.tick_params(axis='both', labelsize=16)

    # # Add thick black y=0 line
    # plt.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.grid(True)

    # Cap y-axis at 1
    X_RANGE = (0, 0.5) if args.experiment == "noise" else (0, 0.2)
    Y_RANGE = (-0.5, 1) if args.experiment == "noise" else (-1, 1)
    # plt.ylim(bottom=Y_RANGE[0], top=Y_RANGE[1])
    # ax.set_xlim(left=X_RANGE[0], right=X_RANGE[1])
    ax.set_ylim(bottom=Y_RANGE[0], top=Y_RANGE[1])

    # plt.legend(loc="lower right", fontsize=14)
    # plt.grid(True)
    # plt.tight_layout()

# fig.supxlabel(X_LABEL, fontsize=20)
# fig.supylabel(Y_LABEL, fontsize=20)

# Legend only once
handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1),  ncol=2, fontsize=16)
fig.legend(handles, labels, loc='lower center',  ncol=2, fontsize=16)

if args.experiment == "optimized":
    fig.tight_layout(rect=[0, 0.1, 1, 1])
else:
    fig.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend at bottom
plt.savefig(f"./paper/figures/test_{args.experiment}.png", dpi=300, bbox_inches='tight')

# plot_file_path = f"./figures/{args.experiment}_{args.model}.png"
# plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
