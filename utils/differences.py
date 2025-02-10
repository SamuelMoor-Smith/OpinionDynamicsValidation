import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon

def calculate_mean_std(true, datasets, type, method="wasserstein"):
    """
    Calculate the mean and standard deviation of the differences between the true dataset and the given datasets."""
    diffs = [dataset_difference(true, d, method=method) for d in datasets]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    print(f"{type} score: {mean_diff} +/- {std_diff}")
    return mean_diff, std_diff

def snapshot_difference(s1, s2, method='wasserstein'):
    """
    Compare two snapshots using the given method.
    """
    if method == 'sorted':
        return difference_sorted(s1, s2)
    elif method == 'wasserstein':
        return difference_wasserstein(s1, s2)
    elif method == 'js':
        return difference_js(s1, s2)
    elif method == 'ks':
        return difference_ks(s1, s2)
    elif method == 'bhattacharyya':
        return difference_bhattacharyya(s1, s2)
    elif method == 'cross_entropy_symmetric':
        return difference_cross_entropy_symmetric(s1, s2)
    else:
        raise ValueError(f"Method {method} not recognized.")

def dataset_difference(d1, d2, method='wasserstein'):
    """
    Compare two datasets using the given method.
    """
    # for s1, s2 in zip(d1.get_data(), d2.get_data()):
    #     print(snapshot_difference(s1, s2, method))
    return sum(snapshot_difference(s1, s2, method) for s1, s2 in zip(d1.get_data(), d2.get_data()))

def create_histograms(x, y, bins=100, range=(0, 1), add_epsilon=False, epsilon=1e-10):
    """
    Create histograms for the given data.
    """
    x_hist, _ = np.histogram(x, bins=bins, range=range, density=True)
    y_hist, _ = np.histogram(y, bins=bins, range=range, density=True)

    # Add epsilon to avoid log(0)
    if add_epsilon:
        x_hist = x_hist + epsilon
        y_hist = y_hist + epsilon

    # renormalize
    x_hist = x_hist / np.sum(x_hist)
    y_hist = y_hist / np.sum(y_hist)

    return x_hist, y_hist

def difference_sorted(x, y):
    """
    Compare the distributions by sorting the values from greatest to smallest
    and then calculating the sum of absolute differences.
    """
    # Sort both x and y in descending order
    x_sorted = np.sort(x)[::-1]
    y_sorted = np.sort(y)[::-1]

    # Calculate absolute differences between the sorted values
    return np.sum(np.abs(x_sorted - y_sorted))

def difference_wasserstein(x, y):
    """
    Compare distributions using Wasserstein distance (Earth Mover's Distance).
    Normalize distributions to [0, 1] before comparison.
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return wasserstein_distance(x, y)

def difference_js(x, y, bins=100, range=(0, 1)):
    """
    Compare distributions using Jensen-Shannon Divergence.
    """
    x_hist, y_hist = create_histograms(x, y, bins=bins, range=range, add_epsilon=True)

    return jensenshannon(x_hist, y_hist)

def difference_ks(x, y):
    """
    Compare distributions using Kolmogorov-Smirnov (KS) test. Requires cdf of both distributions..
    """
    stat, _ = ks_2samp(x, y)
    return stat

def difference_bhattacharyya(x, y, bins=100, range=(-1, 1)):
    """
    Compare distributions using Bhattacharyya distance.
    """
    x_hist, y_hist = create_histograms(x, y, bins=bins, range=range, add_epsilon=True)
    
    # Calculate Bhattacharyya coefficient
    bc = np.sum(np.sqrt(x_hist * y_hist))
    
    # Return Bhattacharyya distance
    return -np.log(bc)

def difference_cross_entropy_symmetric(x, y, bins=100, range=(0, 1)):
    """
    Compare distributions using symmetric cross-entropy.
    """
    x_hist, y_hist = create_histograms(x, y, bins=bins, range=range, add_epsilon=True)

    # Calculate cross-entropy in both directions
    cross_entropy_xy = -np.sum(x_hist * np.log(y_hist))
    cross_entropy_yx = -np.sum(y_hist * np.log(x_hist))
    
    # Return the symmetric cross-entropy
    return (cross_entropy_xy + cross_entropy_yx) / 2