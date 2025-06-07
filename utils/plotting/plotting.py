import matplotlib.pyplot as plt
import os
from datasets.dataset import Dataset
import time
import numpy as np
import seaborn as sns
import pandas as pd
from datasets.ess.header_info import ess_header_info
from models.model import Model
from utils.plotting.plotting_utils import get_yx_fit_y_lower_upper, calculate_explained_variance
import json

def produce_figure(model, filepath, experiment):

    # Get x parameter
    x_param = "noise" if experiment == "noise" else "opinion_drift"
    method = "baseline" if experiment == "reproducibility" else "optimizer"
    y_param = f"explained_variance_{method}"

    with open(filepath, "r") as f:
        json_data = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(json_data)

    # Compute explained variance for optimizer
    df = calculate_explained_variance(df, method=method)

    # Get the fit
    xfit, yfit, ylower, yupper = get_yx_fit_y_lower_upper(df, experiment, x_param, y_param)

    fig, ax = plt.subplots(figsize=(12, 10))

    model_plotting_info = Model.get_model_plotting_info()[model]

    COLOR = model_plotting_info[1]
    DATA_TYPE = "Reproduced" if experiment == "reproducibility" else "Optimized"
    FIT_LABEL = f"{DATA_TYPE} Exponential Fit" if experiment == "noise" else f"{DATA_TYPE} Logarithmic Fit"
    RAW_LABEL = f"{DATA_TYPE} Raw Data"

    ax.scatter(df[x_param], df[y_param], alpha=0.2, label=RAW_LABEL, color=COLOR)
    ax.plot(xfit, yfit, label=FIT_LABEL, color=COLOR)
    ax.fill_between(xfit, ylower, yupper, alpha=0.2, color=COLOR)

    if experiment == "optimized":

        # Compute explained variance for baseline
        df = calculate_explained_variance(df, method="baseline")
        xfit_base, yfit_base, ylower_base, yupper_base = get_yx_fit_y_lower_upper(df, experiment, x_param, "explained_variance_baseline")

        ax.scatter(df[x_param], df["explained_variance_baseline"], alpha=0.2, label="Reproduced Raw Data", color="#808080")
        ax.plot(xfit_base, yfit_base, label="Reproduced Logarithmic Fit", color="#808080")
        ax.fill_between(xfit_base, ylower_base, yupper_base, alpha=0.2, color="#808080")

    # Labels and legend
    TITLE = f"{model_plotting_info[0]} {experiment.capitalize()}"
    if experiment == "noise":
        TITLE = f"{model_plotting_info[0]} with Noise"
    Y_LABEL = "Explained Variance"
    X_LABEL = "Noise" if experiment == "noise" else "Opinion Drift"

    ax.set_title(TITLE, fontsize=24)
    plt.xlabel(X_LABEL, fontsize=22)
    plt.ylabel(Y_LABEL, fontsize=22)
    ax.tick_params(axis='both', labelsize=16)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.grid(True)

    # Cap y-axis at 1
    # X_RANGE = (0, 0.5) if experiment == "noise" else (0, 0.2)
    Y_RANGE = (-0.5, 1) if experiment == "noise" else (-1, 1)
    # ax.set_xlim(left=X_RANGE[0], right=X_RANGE[1])
    ax.set_ylim(bottom=Y_RANGE[0], top=Y_RANGE[1])

    plt.legend(loc="lower right", fontsize=14)
    plt.tight_layout()
    image_filepath = filepath.replace(".jsonl", ".png")
    plt.savefig(image_filepath, dpi=300, bbox_inches='tight')


def produce_stripplot():

    model_info = Model.get_model_plotting_info()

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


def plot_2_datasets_snapshots(d1: Dataset, d2: Dataset, path):
    """
    Plots snapshots of two datasets side by side with histograms per time step.
    
    Parameters:
        d1, d2: Dataset objects with get_data(), get_opinion_range(), get_params()
        path: optional directory to save the figure
        bins: number of bins for histogram
        filename: optional filename for saved image
        difference: metric name for computing snapshot differences (e.g. 'wasserstein')
    """
    data1 = d1.get_data()
    data2 = d2.get_data()
    op_range = d1.get_opinion_range()
    n_snapshots = len(data1)

    rows, cols = 3, 3

    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    name = d2.model.get_model_name().capitalize()
    fig.suptitle(
        f"{name} Model: Optimized vs. Ground-Truth Over Time",
        fontsize=24
    )

    # Compute max y-value per row for consistent scaling
    row_y_max = [0 for _ in range(rows)]
    for i in range(9):
        row = i // cols
        h1, _ = np.histogram(data1[i], bins=100, range=op_range)
        h2, _ = np.histogram(data2[i], bins=100, range=op_range)
        row_y_max[row] = max(row_y_max[row], h1.max(), h2.max())

    for i, ax in enumerate(axes.flat[:9]):
        ax.hist(data1[i], bins=100, range=op_range, alpha=0.5, label='Data1')
        ax.hist(data2[i], bins=100, range=op_range, alpha=0.5, label='Data2')

        ax.set_title(f'Round {i+1}', fontsize=16)
        ax.set_xlim(*op_range)
        ax.set_ylim(0, row_y_max[i // cols])
        ax.tick_params(axis='both', labelsize=12)

        # Hide redundant ticks
        if i // cols < rows - 1:
            ax.set_xticklabels([])
        if i % cols != 0:
            ax.set_yticklabels([])

    # Global axis labels
    fig.supxlabel("Opinion Value", fontsize=18)
    fig.supylabel("Frequency", fontsize=18)

    # plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title
    # ---- Add full-width bottom legend with model params ----
    param_str_1 = f"Ground Truth Params:\n{d1.get_params()}"
    param_str_2 = f"Optimized Params:\n{d2.get_params()}"
    full_legend_text = f"{param_str_1}\n\n{param_str_2}"

    # Add the full-width text box at the bottom
    fig.text(
        0.5, -0.04, full_legend_text,
        ha='center', va='top',
        fontsize=10, wrap=True, family='monospace'
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # leave room for the bottom text and title

    os.makedirs(path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"d1_vs_d2_snapshots_{timestamp}.png"
    plt.savefig(os.path.join(path, filename), dpi=300, bbox_inches="tight")
    plt.close()
