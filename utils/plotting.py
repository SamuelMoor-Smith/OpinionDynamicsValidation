import matplotlib.pyplot as plt
import os
from datasets.dataset import Dataset
from utils.differences import snapshot_difference
import time
import math
import numpy as np

def plot_2_snapshots(
        s1,
        s2,
        difference="wasserstein",
        path=None,
        bins=100,
        filename=None
):
    """plots 2 snapshots side by side"""

    scores = snapshot_difference(s1, s2, method=difference)

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Snapshot 1 and Snapshot 2\n{difference} Score: {scores:.3f}")

    # Plot data 1
    plt.subplot(1, 2, 1)
    plt.hist(s1, bins=bins, edgecolor='black', alpha=0.7)
    plt.title('Snapshot 1')
    plt.xlabel('Opinion Value')
    plt.ylabel('Frequency')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    # Plot data 2
    plt.subplot(1, 2, 2)
    plt.hist(s2, bins=bins, edgecolor='black', alpha=0.7)
    plt.title('Snapshot 2')
    plt.xlabel('Opinion Value')
    plt.ylabel('Frequency')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.tight_layout(pad=1.5)  # Adjusted padding for better fit

    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"s1_vs_s2_{timestamp}.png"
        plt.savefig(os.path.join(path, filename))
        # print(f"Plot saved at {os.path.join(save_path, filename)}")
        plt.close()  # Close the plot to free up memory
    else:
        plt.show()

def plot_2_datasets_snapshots(
        d1: Dataset,
        d2: Dataset,
        difference="wasserstein",
        path=None,
        bins=100,
        filename=None
):
    """plots the 2 datasets snapshots side by side"""

    data1 = d1.get_data()
    data2 = d2.get_data()

    print("Parameters: ", d2.get_params())

    N = len(data1)

    cols = 3
    rows = 3

    scores = [snapshot_difference(data1[i], data2[i], method=difference) for i in range(N)]

    # 📌 Step 1: Precompute y_max across all histograms
    op_range = d1.get_opinion_range()
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))  # Set tight layout

    # name = d1.model.__class__.__name__
    # Set a shared title
    fig.suptitle(
        # f"Model 1: {d1.model.__class__.__name__} (Params: {d1.get_params()})\n"
        # f"Model 2: {d2.model.__class__.__name__} (Params: {d2.get_params()})\n"
        # f"Total Score Sum: {sum(scores):.3f}",
        # f"{name[0].upper() + name[1:]}",
        "Deffuant Model: Optimized vs. Ground-Truth Over Time",
        fontsize=24
    )

    row_y_max = [0 for _ in range(rows)]
    for i, ax in enumerate(axes.flat):
        h1, _, _ = ax.hist(data1[i], bins=bins, range=op_range, alpha=0.5, label='Data1')
        h2, _, _ = ax.hist(data2[i], bins=bins, range=op_range, alpha=0.5, label='Data2')
        # if i > 0:
        #     ax.hist(data1[i-1], bins=bins, range=d1.get_opinion_range(), alpha=0.2, label='Data1Z')

        # y_max = max(y_max, h1.max(), h2.max())
        row = i // cols
        row_y_max[row] = max(row_y_max[row], h1.max(), h2.max())

        # if i % cols == cols -1: # Last column
        #     # iterate through the row
        #     for ax2 in axes.flat[i-cols+1:i+1]: 
        #         ax2.set_ylim(bottom=0, top=y_max)

        # Remove axis labels except for bottom row (X) and leftmost column (Y)
        if i // cols < rows - 1:  # Not in the last row
            ax.set_xticklabels([])
        if i % cols != 0:  # Not in the first column
            ax.set_yticklabels([])
         # ax.set_title(f'Score: ', fontsize=8)
        ax.tick_params(axis='both', labelsize=16)

        # # ax.set_title(f'Round {i+1}', fontsize=8)
        # score_base = snapshot_difference(data1[i], data2[i], method=difference)
        # if i == 0:
        #     ax.set_title(f'{score_base:.3f}', fontsize=8)
        # else:
        #     score_zero = snapshot_difference(data1[i], data1[i-1], method=difference)
        ax.set_title(f'Round {i+1}', fontsize=20)
        ax.set_xlim(left=op_range[0], right=op_range[1])

    for i, ax in enumerate(axes.flat):
        # Set y-limits for each row
        ax.set_ylim(bottom=0, top=row_y_max[i // cols])

    # Shared X and Y labels
    fig.supxlabel("Opinion Value", fontsize=22)
    fig.supylabel("Frequency", fontsize=22)

    plt.tight_layout()
    # plt.xlim(op_range[0], op_range[1])
    # plt.ylim(0, y_max)  

    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if not filename:
            filename = f"d1_vs_d2_snapshots_{timestamp}.png"
        plt.savefig(os.path.join(path, filename), dpi=300, bbox_inches="tight")
        plt.close()  # Close the plot to free up memory
    else:
        plt.show()