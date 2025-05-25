import numpy as np
from models.model import Model
from scipy.stats import beta
    
class DistortionAdaptor(Model):

    TRANSFORM_PARAM_RANGES = {
        'a': (0.1, 10),
        'b': (0.1, 10)
    }

    def __init__(self, base_model: Model, params=None, seed=None):

        self.base_model = base_model
        self.transform = BetaCDFTransformation()
        
        base_class = base_model.__class__
        DistortionAdaptor.MODEL_NAME = f"distorted_{base_class.get_model_name()}"
        DistortionAdaptor.OPINION_RANGE = base_class.get_opinion_range()
        DistortionAdaptor.PARAM_RANGES = {**base_model.PARAM_RANGES, **self.TRANSFORM_PARAM_RANGES}

        super().__init__(params=params, seed=seed)

    def run(self, input, p=None):

        p = self.params if p is None else p

        # Extract transform params
        a, b = p['a'], p['b']
        transformed_input = self.transform.forward(input, a, b)

        # Extract base model params
        model_params = {k: v for k, v in p.items() if k not in {'a', 'b'}}
        self.base_model.params = model_params
        output = self.base_model.run(transformed_input, model_params)
        
        return self.transform.backward(output, a, b)
    
class Transformation:

    def assert_valid(self, x, **kwargs):
        """
        Assert that the input is valid.
        """
        assert x.min() >= 0, "Input values must be non-negative."
        assert x.max() <= 1, "Input values must be less than or equal to 1."

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
    def assert_valid(self, x, a, b, c):
        super().assert_valid(x)
        assert a > 0, "Parameter 'a' must be positive."
        assert c > 0, "Parameter 'c' must be positive."
        return x.clip(1e-8, 1-1e-8)  # Floating point precision results in 1 sometimes and this causes problems
    
    def forward(self, input, a, b, c):
        input = self.assert_valid(input, a, b, c)
        return ( 1 / (1 + np.exp(-a * (input - b))) ) ** c
    
    def backward(self, output, a, b, c):
        output = self.assert_valid(output, a, b, c)
        tmp_output = output ** (1 / c)
        return (1 / a) * np.log(tmp_output / (1 - tmp_output)) + b
    
class BetaCDFTransformation(Transformation):
    """
    Transformation using the Beta CDF and its inverse.
    """

    def assert_valid(self, x, a, b):
        super().assert_valid(x)
        assert a > 0, "Parameter 'a' must be positive."
        assert b > 0, "Parameter 'b' must be positive."
        return x

    def forward(self, input, a, b):
        input = self.assert_valid(input, a, b)
        return beta.cdf(input, a, b)

    def backward(self, output, a, b):
        output = self.assert_valid(output, a, b)
        return beta.ppf(output, a, b)
