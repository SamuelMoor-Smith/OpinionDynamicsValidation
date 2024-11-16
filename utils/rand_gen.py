import numpy as np

def generate_multiple_random_nums(max_val, N):
    """
    Generate multiple random numbers.
    """
    return np.random.randint(0, max_val, int(N))

def generate_multiple_random_pairs(max_val, num_pairs):
    """
    Generate multiple random pairs. Possible for pair to have the same num twice.
    """
    return np.random.randint(0, max_val, (int(num_pairs), 2))

def create_random_opinion_distribution(N=1000, min_val=-1, max_val=1, seed=42):
    # Generate random initial opinions (between -1 and 1) for N agents
    np.random.seed(seed)  # Set a seed value
    ops = np.random.uniform(min_val, max_val, N)  # Random distribution between min_val and max_val
    np.random.seed(None)  # Reset the seed to None to return randomness to the system clock or entropy source
    return ops
