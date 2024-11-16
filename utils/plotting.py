import matplotlib.pyplot as plt
import os
from datasets.dataset import Dataset
from utils.differences import snapshot_difference
import time
import math

def plot_2_datasets_snapshots(
        d1: Dataset,
        d2: Dataset,
        difference="wasserstein",
        path=None,
        bins=100,
):
    """plots the 2 datasets snapshots side by side"""

    data1 = d1.get_data()
    data2 = d2.get_data()

    N = len(data1)

    cols = 4
    rows = math.ceil(N*2/cols)

    scores = [snapshot_difference(data1[i], data2[i], method=difference) for i in range(N)]

    plt.figure(figsize=(15, 15)) 
    plt.suptitle(
        f"""
            Model 1: {d1.model.__class__.__name__} and Params: {d1.model.params}
            Model 2: {d2.model.__class__.__name__} and Params: {d2.model.params}
            Total score sum:{sum(scores):.3f}
        """
    )
    
    for i in range(N):
        # Plot true data
        plt.subplot(rows, cols, i * 2 + 1)
        plt.hist(data1[i], bins=bins, range=d1.get_opinion_range(), edgecolor='black', alpha=0.7)
        plt.title(f'Data1 Snapshot {i+1}')
        plt.xlabel('Opinion Value')
        plt.ylabel('Frequency')
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        # Plot simulated data (model run results)
        plt.subplot(rows, cols, i * 2 + 2)
        plt.hist(data2[i], bins=bins, range=d2.get_opinion_range(), edgecolor='black', alpha=0.7)
        plt.title(f'Data2 Snapshot {i+1}\n{difference} Score: {scores[i]:.3f}')
        plt.xlabel('Opinion Value')
        plt.ylabel('Frequency')
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.tight_layout(pad=1.5)  # Adjusted padding for better fit

    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"d1_vs_d2_snapshots_{timestamp}.png"
        plt.savefig(os.path.join(path, filename))
        # print(f"Plot saved at {os.path.join(save_path, filename)}")
        plt.close()  # Close the plot to free up memory
    else:
        plt.show()
