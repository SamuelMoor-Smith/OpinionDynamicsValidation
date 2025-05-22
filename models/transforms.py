import numpy as np
import warnings

# sigmoid with x^alpha
# combine both of them - 2 or 3 parameters
# sigmoid(a,b)^c (c > 1 or c < 1)
# ranges of the parameters
# renormalized

# test if model without distortions can reconstruct one with distortions
# want answer to be no

class Transformation:

    def assert_valid(self, x, a, b, c):
        """
        Assert that the input is valid.
        """
        assert x.min() >= 0, "Input values must be non-negative."
        assert x.max() <= 1, "Input values must be less than or equal to 1."
        assert a > 0, "Parameter 'a' must be positive."
        assert c > 0, "Parameter 'c' must be positive."
        return x.clip(1e-8, 1-1e-8)  # Floating point precision results in 1 sometimes and this causes problems

    def forward(self, input, a, b, c):
        """
        Forward transformation function.
        """
        raise NotImplementedError("Forward transformation not implemented.")

    def backward(self, output, a, b, c):
        """
        Backward transformation function.
        """
        raise NotImplementedError("Backward transformation not implemented.")
    
class SigmoidTransformation(Transformation):
    """
    Sigmoid transformation with exponentiation.
    
    -- sigmoid(a, b)^c
    """
    
    def forward(self, input, a, b, c):
        input = super().assert_valid(input, a, b, c)
        return ( 1 / (1 + np.exp(-a * (input - b))) ) ** c
    
    def backward(self, output, a, b, c):
        output = super().assert_valid(output, a, b, c)
        tmp_output = output ** (1 / c)
        return (1 / a) * np.log(tmp_output / (1 - tmp_output)) + b
    
# class PowerLawTransformation(Transformation):
#     """
#     Power law transformation.
#     """
    
#     def forward(self, input, a, b, c):
#         super().assert_valid(input)
#         return input ** a
    
#     def backward(self, output, a, b, c):
#         super().assert_valid(output)
#         return output ** (1 / a)
    
# class LogitTransformation(Transformation):
#     """
#     Logistic growth-like transformation.
#     """
    
#     def forward(self, input, a, b, c):
#         super().assert_valid(input)
#         return input**a / (input**a + (1 - input)**a)
    
#     def backward(self, output, a, b, c):
#         super().assert_valid(output)
#         tmp_output = (output / (1 - output)) ** (1/a)
#         return tmp_output / (1 + tmp_output)