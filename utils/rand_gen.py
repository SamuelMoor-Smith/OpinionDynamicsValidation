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

def create_random_opinion_distribution(N=1000, min_val=0, max_val=1, seed=None):
    """Generate random initial opinions for N agents."""

    # Set a seed value for reproducibility
    np.random.seed(seed)

    # Random distribution between min_val and max_val
    ops = np.random.uniform(min_val, max_val, N)

    # Reset the seed to None to return randomness to the system clock or entropy source
    np.random.seed(None) 
    
    return ops
