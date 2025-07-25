import numpy as np
from numba import njit

def add_noise(input, output, noise, model):
    """
    Add noise to the outputs.
    """
    if noise < 0:
        raise ValueError("Noise parameter must be non-negative.")

    variance_basic = variance(input, output)
    noisy_output = output + np.random.normal(0, noise, len(output))
    keep_values_in_range(noisy_output, model)
    variance_noisy = variance(input, noisy_output)

    return noisy_output, variance_basic, variance_noisy

def variance(input, output):
    """
    Calculate the variance of the output.
    """
    return np.sum((output - input) ** 2) 

@njit
def clip_value_in_range(value, min_val, max_val):
    """
    Keep a value in the opinion range [min_val, max_val] by reflecting out-of-range values back into the range.
    """

    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value

@njit
def keep_value_in_range(value, min_val, max_val):
    """
    Keep a value in the opinion range [min_val, max_val] by reflecting out-of-range values back into the range.
    """

    if value > min_val and value < max_val:
        return value

    res = value
    if value < min_val:
        res = (min_val - value) + min_val
    if value > max_val:
        res = max_val - (value - max_val)

    assert min_val <= res <= max_val, "Value out of range after clipping."

    return res

def keep_values_in_range(values, model):
    """
    Keep values in the opinion range [min_val, max_val] by reflecting out-of-range values back into the range.
    """
    if model is None:
        min_val = -1
        max_val = 1
    else:
        min_val, max_val = model.get_opinion_range()

    values[values < min_val] = \
        (min_val - values[values < min_val]) + min_val
    values[values > max_val] = \
        max_val - (values[values > max_val] - max_val)
    return values