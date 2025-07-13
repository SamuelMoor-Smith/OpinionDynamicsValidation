import numpy as np
from models.model import Model
from scipy.stats import beta
import matplotlib.pyplot as plt
import random
import os

def plot_distortion(transformation, a, b, title="Distortion", show_inverse=False):
    x = np.linspace(0, 1, 500)
    y = transformation.forward(x, a, b)

    plt.figure(figsize=(6, 4))
    plt.plot(x, x, '--', label='Identity (no distortion)', color='gray')
    plt.plot(x, y, label=f'Forward (a={a}, b={b})', color='blue')

    if show_inverse:
        try:
            inv_x = np.linspace(0, 1, 500)
            inv_y = transformation.backward(inv_x, a, b)
            plt.plot(inv_x, inv_y, label='Inverse (Backward)', color='green')
        except Exception as e:
            print("Could not plot inverse:", e)

    plt.xlabel("Input")
    plt.ylabel("Transformed Output")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    # Create output directory if it doesn't exist
    save_dir = "results/distortions"
    os.makedirs(save_dir, exist_ok=True)

    # Construct filename
    filename = f"distortion_a{a:.2f}_b{b:.2f}.png"
    filepath = os.path.join(save_dir, filename)

    # Save plot to file
    plt.savefig(filepath)
    plt.close()  # Close the figure to avoid memory issues in loops

    print(f"Saved: {filepath}")
    
class DistortionAdaptor(Model):

    TRANSFORM_PARAM_RANGES = {
        'a': (0.5, 2),
        'b': (0.5, 2)
    }

    def __init__(self, base_model: Model, params=None, seed=None):

        self.base_model = base_model
        
        base_class = base_model.__class__
        DistortionAdaptor.MODEL_NAME = f"distorted_{base_class.get_model_name()}"
        DistortionAdaptor.OPINION_RANGE = base_class.get_opinion_range()
        DistortionAdaptor.PARAM_RANGES = {**base_model.PARAM_RANGES, **self.TRANSFORM_PARAM_RANGES}

        self.transform = BetaCDFTransformation(DistortionAdaptor.OPINION_RANGE[0], DistortionAdaptor.OPINION_RANGE[1])

        super().__init__(params=params, seed=seed)

        # plot_distortion(self.transform, a=self.params["a"], b=self.params["b"], title="Beta CDF Distortion", show_inverse=True)

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

    def _complete_params(self, partial_params):
        """Fill in missing parameters with random values."""
        if self.PARAM_RANGES is None:
            raise NotImplementedError("PARAM_RANGES must be defined in subclass.")

        np.random.seed(self.seed)
        complete = {}

        for name, (low, high) in self.PARAM_RANGES.items():
            if partial_params and name in partial_params:
                complete[name] = partial_params[name]
            elif name == 'a' or name == 'b':
                rand1 = np.random.uniform(low, 1)
                rand2 = np.random.uniform(1, high)

                complete[name] = rand1 if random.random() < 0.5 else rand2
            else:
                complete[name] = np.random.uniform(low, high)

        return complete
    
class Transformation:

    @staticmethod
    def assert_valid(x, **kwargs):
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
    
# class SigmoidTransformation(Transformation):
#     """
#     Sigmoid transformation with exponentiation.
    
#     -- sigmoid(a, b)^c
#     """
#     def assert_valid(self, x, a, b, c):
#         super().assert_valid(x)
#         assert a > 0, "Parameter 'a' must be positive."
#         assert c > 0, "Parameter 'c' must be positive."
#         return x.clip(1e-8, 1-1e-8)  # Floating point precision results in 1 sometimes and this causes problems
    
#     def forward(self, input, a, b, c):
#         input = self.assert_valid(input, a, b, c)
#         return ( 1 / (1 + np.exp(-a * (input - b))) ) ** c
    
#     def backward(self, output, a, b, c):
#         output = self.assert_valid(output, a, b, c)
#         tmp_output = output ** (1 / c)
#         return (1 / a) * np.log(tmp_output / (1 - tmp_output)) + b
    
class BetaCDFTransformation(Transformation):
    """
    Transformation using the Beta CDF and its inverse.
    """

    def __init__(self, op_min, op_max):
        """
        Initialize the transformation with the opinion range.
        """
        self.op_min = op_min
        self.op_max = op_max

    def assert_valid(self, x, a, b):
        super().assert_valid(x)
        assert a > 0, "Parameter 'a' must be positive."
        assert b > 0, "Parameter 'b' must be positive."
        return x

    def normalize(self, x):
        """
        Normalize the input to the range [0, 1].
        """
        assert x.min() >= self.op_min, "Input values must be greater than or equal to op_min."
        assert x.max() <= self.op_max, "Input values must be less than or equal to op_max."
        op_var = self.op_max - self.op_min
        # print(self.op_min, self.op_max, op_var)
        return (x - self.op_min) / op_var
    
    def denormalize(self, x):
        """
        Denormalize the input from the range [0, 1] back to the original range.
        """
        op_var = self.op_max - self.op_min
        return x * op_var + self.op_min

    def forward(self, input, a, b):

        input = self.normalize(input)

        input = self.assert_valid(input, a, b)
        res = beta.cdf(input, a, b)

        return self.denormalize(res)  # Scale back to original range for model execution

    def backward(self, output, a, b):
        
        output = self.normalize(output)

        output = self.assert_valid(output, a, b)
        res = beta.ppf(output, a, b)

        return self.denormalize(res)  # Scale back to original range for model results
