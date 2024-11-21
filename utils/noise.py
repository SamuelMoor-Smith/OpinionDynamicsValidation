import numpy as np

def add_noise(input, output, noise):
    """
    Add noise to the outputs.
    """
    if noise < 0:
        raise ValueError("Noise parameter must be non-negative.")

    variance_basic = variance(input, output)
    noisy_output = output + np.random.normal(0, noise, len(output))
    keep_in_range(noisy_output)
    variance_noisy = variance(input, noisy_output)

    return noisy_output, variance_basic, variance_noisy

def variance(input, output):
    """
    Calculate the variance of the output.
    """
    return np.sum((output - input) ** 2) 

def keep_in_range(values):
    """
    Keep values in the opinion range [0, 1] by reflecting out-of-range values back into the range.
    """
    values[values < 0] = -values[values < 0]
    values[values > 1] = 2 - values[values > 1]
    return values