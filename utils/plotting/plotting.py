import matplotlib.pyplot as plt
import os
from datasets.dataset import Dataset
import time
import numpy as np
import seaborn as sns
import pandas as pd
from datasets.ess.header_info import ess_header_info
from models.model import Model
from utils.plotting.plotting_utils import get_yx_fit_y_lower_upper, calculate_explained_variance, calculate_explained_variance_real
import json

sns.set_style("whitegrid")
sns.set_context("talk")  # larger fonts

FAMILY = "sans-serif"

def produce_figure(generator, predictor, filepath, experiment):

    # Get x parameter
    x_param = "noise" if experiment == "noise" else "opinion_drift"
    method = "baseline" if experiment == "reproducibility" else "optimizer"
    y_param = f"explained_variance_{method}"

    with open(filepath, "r") as f:
        json_data = [
            json.loads(line)
            for line in f
            if line.strip() and not line.lstrip().startswith("//")
        ]
    df = pd.DataFrame(json_data)

    # Compute explained variance for optimizer
    df = calculate_explained_variance(df, method=method)

    # Get the fit
    xfit, yfit, ylower, yupper = get_yx_fit_y_lower_upper(df, experiment, x_param, y_param)

    fig, ax = plt.subplots(figsize=(12, 10))

    distorted_predictor = "distorted" in predictor
    distorted_generator = "distorted" in generator

    predictor_plotting_info = Model.get_model_plotting_info()[predictor.removeprefix("distorted_")]
    generator_plotting_info = Model.get_model_plotting_info()[generator.removeprefix("distorted_")]

    COLOR = predictor_plotting_info[1]
    DATA_TYPE = "Generator" if experiment == "reproducibility" else "Predictor"
    FIT_LABEL = f"{DATA_TYPE} Exponential Fit" if experiment == "noise" else f"{DATA_TYPE} Logarithmic Fit"
    RAW_LABEL = f"{DATA_TYPE} Raw Data"

    ax.scatter(df[x_param], df[y_param], alpha=0.2, label=RAW_LABEL, color=COLOR, rasterized=True)
    if distorted_predictor:
        ax.plot(xfit, yfit, label=FIT_LABEL, color=COLOR, linestyle="--")
    else:
        ax.plot(xfit, yfit, label=FIT_LABEL, color=COLOR)
    ax.fill_between(xfit, ylower, yupper, alpha=0.2, color=COLOR)

    if experiment == "optimized":

        # Compute explained variance for baseline
        df = calculate_explained_variance(df, method="baseline")
        xfit_base, yfit_base, ylower_base, yupper_base = get_yx_fit_y_lower_upper(df, experiment, x_param, "explained_variance_baseline")

        ax.scatter(df[x_param], df["explained_variance_baseline"], alpha=0.2, label="Generator Raw Data", color="#808080", rasterized=True)
        if distorted_generator:
            ax.plot(xfit_base, yfit_base, label="Generator Logarithmic Fit", color="#808080", linestyle="--")
        else:
            ax.plot(xfit_base, yfit_base, label="Generator Logarithmic Fit", color="#808080")
        ax.fill_between(xfit_base, ylower_base, yupper_base, alpha=0.2, color="#808080")

    # Labels and legend
    # TITLE = f"{model_plotting_info[0]} {experiment.capitalize()}"
    generator_name = generator_plotting_info[0] # .strip(" Model")
    predictor_name = predictor_plotting_info[0].strip(" Model")

    # print(generator_name, predictor_name)
    # if "distorted" in generator:
    #     generator_name = f"Distorted {generator_name}"
    # if "distorted" in predictor:
    #     predictor_name = f"Distorted {predictor_name}"

    # # TITLE = f"Generator Model: {generator_name}\nPredictor Model: {predictor_name}"
    # generator_label = "Generator"
    # predictor_label = "Predictor"

    # label_width = max(len(generator_label), len(predictor_label))
    # model_width = max(len(generator_name), len(predictor_name))
    # TITLE = "\n".join([
    #     f"{generator_label.rjust(label_width)}: {generator_name.ljust(model_width)}",
    #     f"{predictor_label.rjust(label_width)}: {predictor_name.ljust(model_width)}",
    # ])
    TITLE = generator_name

    if experiment == "noise":
        TITLE = TITLE = f"Generator Model: {generator_name} with Noise\nPredictor Model: {predictor_name}"

    Y_LABEL = "Explained Variance"
    X_LABEL = "Noise" if experiment == "noise" else "Opinion Drift"

    ax.set_title(TITLE, fontsize=50, fontfamily=FAMILY, fontweight='medium', pad=20)
    plt.xlabel(X_LABEL, fontsize=40, fontfamily=FAMILY)
    plt.xlim(left=0)
    show_y_axis = "deffuant" in TITLE.lower() and "repulsion" not in TITLE.lower()
    if show_y_axis:
        plt.ylabel(Y_LABEL, fontsize=40, fontfamily=FAMILY)
    else:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=30)
    if show_y_axis:
        ax.tick_params(axis='y', labelsize=30)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.grid(True)

    # Cap y-axis at 1
    # X_RANGE = (0, 0.5) if experiment == "noise" else (0, 0.2)
    Y_RANGE = (-0.5, 1) if experiment == "noise" else (-1, 1)
    # ax.set_xlim(left=X_RANGE[0], right=X_RANGE[1])
    ax.set_ylim(bottom=Y_RANGE[0], top=Y_RANGE[1])

    ax.legend(loc="lower right", prop={"family": FAMILY, "size": 26})
    plt.tight_layout()
    image_filepath = filepath.replace(".jsonl", ".png")
    plt.savefig(image_filepath, dpi=200, bbox_inches='tight')
    # plt.savefig(
    #     image_filepath,
    #     bbox_inches='tight'
    # )
    print(f"Saved figure to {image_filepath}")


def produce_stripplot(distorted=False):

    sns.set_context("notebook")

    model_info = Model.get_model_plotting_info()

    # Create palette based on model_info
    palette = {info[0]: info[1] for info in model_info.values()}

    dfs = []
    for model in model_info.keys():
        if distorted:
            df = pd.read_json(f"results/real/distorted_{model}_3.jsonl", lines=True)
        else:
            df = pd.read_json(f"results/real/{model}_3.jsonl", lines=True)
        df["Model Type"] = model_info[model][0]
        df = calculate_explained_variance_real(df)
        dfs.append(df)

    # Combine all DataFrames
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined["Dataset"] = df_combined["ess_key"].map(lambda k: f"{k}-{ess_header_info[k]['country'][:2].upper()}")

    # Plot

    if distorted:
        TITLE = f"Distorted Models Performance on ESS Data"
    else:
        TITLE = f"Previous Models Performance on ESS Data"
    Y_LABEL = "Explained Variance"
    X_LABEL = "ESS Dataset"

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.stripplot(
        data=df_combined, 
        x="Dataset", 
        y="explained_variance", 
        hue="Model Type", 
        dodge=True, 
        alpha=0.5, 
        palette=palette, 
        legend=not distorted,
        size=5,
        ax=ax)
    
    sns.pointplot(
        data=df_combined,
        x="Dataset",
        y="explained_variance",
        hue="Model Type",
        dodge=0.64, 
        join=False,
        markers="D",
        scale=1.0,
        ci="sd",  # standard deviation; use "se" for standard error, or None for no bars
        # palette=palette,
        color='black',  # Use a single color for the pointplot
        errwidth=2,
        zorder=10,
        legend=False,
        ax=ax
    )

    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(legend.get_title().get_text(), prop={"family": FAMILY, "size": 14})
        for text in legend.get_texts():
            text.set_fontfamily(FAMILY)
            text.set_fontsize(12)

    plt.axhline(y=0, color='black', linewidth=2)

    plt.title(TITLE, fontsize=20, fontfamily=FAMILY)

    plt.ylabel(Y_LABEL, fontsize=18, fontfamily=FAMILY)
    plt.xlabel(X_LABEL, fontsize=18, fontfamily=FAMILY)
    plt.xticks(rotation=45, fontsize=16, fontfamily=FAMILY)
    plt.yticks(fontsize=16, fontfamily=FAMILY)

    plt.ylim(-0.7, 0.3)

    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if distorted:
        SAVE_FILE = f"results/figures/stripplot_distorted_{timestamp}.pdf"
    else:
        SAVE_FILE = f"results/figures/stripplot_{timestamp}.pdf"
    plt.savefig(SAVE_FILE, bbox_inches="tight")
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
        fontsize=24,
        fontfamily=FAMILY
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

        # sns.histplot(data1[i], bins=100, binrange=op_range, ax=ax, kde=False, color='blue', label='Data1', alpha=0.5, element="step")
        # sns.histplot(data2[i], bins=100, binrange=op_range, ax=ax, kde=False, color='orange', label='Data2', alpha=0.5, element="step")

        ax.set_title(f'Round {i+1}', fontsize=16, fontfamily=FAMILY)
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
        fontsize=10, wrap=True, family=FAMILY
    )

    fig.supxlabel("Opinion Value", fontsize=18, fontfamily=FAMILY)
    fig.supylabel("Frequency", fontsize=18, fontfamily=FAMILY)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # leave room for the bottom text and title

    os.makedirs(path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{name}_{timestamp}.png"
    plt.savefig(os.path.join(path, filename), dpi=300, bbox_inches="tight")
    plt.close()

def plot_dataset_snapshots(d: Dataset, path, bins=100):
    """
    Plots snapshots of one dataset with histograms per time step.
    """
    data = d.get_data()
    op_range = d.get_opinion_range()
    n_snapshots = len(data)

    rows, cols = 3, 3

    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    name = d.model.get_model_name().capitalize() if d.model else "Raw Dataset?"
    fig.suptitle(
        f"{name} Model: Optimized vs. Ground-Truth Over Time",
        fontsize=24
    )

    # Compute max y-value per row for consistent scaling
    row_y_max = [0 for _ in range(rows)]
    for i in range(9):
        row = i // cols
        h, _ = np.histogram(data[i], bins=bins, range=op_range)
        row_y_max[row] = max(row_y_max[row], h.max())

    for i, ax in enumerate(axes.flat[:9]):

        ax.hist(data[i], bins=bins, range=op_range, alpha=0.5, label='Data')
        # sns.histplot(data[i], bins=bins, binrange=op_range, ax=ax, kde=False, color='blue', label='Data', alpha=0.5, element="step")

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
    param_str_1 = f"Ground Truth Params:\n{d.get_params()}"
    full_legend_text = f"{param_str_1}"

    # Add the full-width text box at the bottom
    fig.text(
        0.5, -0.04, full_legend_text,
        ha='center', va='top',
        fontsize=10, wrap=True, family=FAMILY
    )

    fig.supxlabel("Opinion Value", fontsize=18, fontfamily=FAMILY)
    fig.supylabel("Frequency", fontsize=18, fontfamily=FAMILY)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # leave room for the bottom text and title

    os.makedirs(path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{name}_{timestamp}.png"
    plt.savefig(os.path.join(path, filename), dpi=300, bbox_inches="tight")
    plt.close()
