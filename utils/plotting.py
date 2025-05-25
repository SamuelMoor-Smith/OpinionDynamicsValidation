import matplotlib.pyplot as plt
import os
from datasets.dataset import Dataset
import time
import numpy as np
import seaborn as sns
import pandas as pd
from datasets.ess.header_info import ess_header_info

def produce_stripplot():

    # Model name mapping
    model_info = {
        "deffuant": ("Deffuant Model", "C0"),
        "hk_averaging": ("HK Averaging Model", "C1"),
        "carpentras": ("ED Model", "C4"),
        "duggins": ("Duggins Model", "C2"),
        "transform_deffuant": ("Transformed Deffuant Model", "C3"),
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
    for i in range(n_snapshots):
        row = i // cols
        h1, _, _ = np.histogram(data1[i], bins=bins, range=op_range)
        h2, _, _ = np.histogram(data2[i], bins=bins, range=op_range)
        row_y_max[row] = max(row_y_max[row], h1.max(), h2.max())

    for i, ax in enumerate(axes.flat[:n_snapshots]):
        ax.hist(data1[i], bins=bins, range=op_range, alpha=0.5, label='Data1')
        ax.hist(data2[i], bins=bins, range=op_range, alpha=0.5, label='Data2')

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

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title

    os.makedirs(path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"d1_vs_d2_snapshots_{timestamp}.png"
    plt.savefig(os.path.join(path, filename), dpi=300, bbox_inches="tight")
    plt.close()
