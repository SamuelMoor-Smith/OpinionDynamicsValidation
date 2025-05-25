import numpy as np
from scipy.stats import wasserstein_distance
from datasets.dataset import Dataset

def calculate_mean_std(true, datasets):
    """
    Calculate the mean and standard deviation of the differences between the true dataset and the given datasets."""
    diffs = [dataset_difference(true, d) for d in datasets]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    print(f"Wasserstein mean loss: {mean_diff} +/- {std_diff}")
    return mean_diff, std_diff

def snapshot_difference(s1, s2, range):
    """
    Compare distributions using Wasserstein distance (Earth Mover's Distance).
    Normalize distributions to [0, 1] before comparison.

    The normalization is done to make sure the scales of the distributions are comparable.
    """
    s1 = (s1 - range[0]) / (range[1] - range[0])
    s2 = (s2 - range[0]) / (range[1] - range[0])
    return wasserstein_distance(s1, s2)

def dataset_difference(d1: Dataset, d2: Dataset):
    """
    Compare two datasets using the given method.
    """
    print(d2.get_opinion_range())
    return sum(snapshot_difference(s1, s2, range=d2.get_opinion_range()) for s1, s2 in zip(d1.get_data(), d2.get_data()))
