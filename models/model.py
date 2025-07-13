from utils.rand_gen import create_random_opinion_distribution
import numpy as np

class Model:

    # Class variables
    MODEL_NAME = None
    OPINION_RANGE = None
    PARAM_RANGES = None

    # Will be set in __init__
    params = None

    def __init__(self, params=None, seed=None):
        self.seed = seed
        self.params = self._complete_params(params)
        print(f"{self.MODEL_NAME} model created with parameters {self.params}")

    def run_main_loop_with_njit(self, opinions, **kwargs):
        """
        Run the model with input and additional keyword arguments.
        This method is intended to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement run_main_loop_with_njit.")
    
    def run(self, input, p=None):
        """
        Run the model with input and parameters.
        This method is intended to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement run.")
    
    def _complete_params(self, partial_params):
        """Fill in missing parameters with random values."""
        if self.PARAM_RANGES is None:
            raise NotImplementedError("PARAM_RANGES must be defined in subclass.")

        np.random.seed(self.seed)
        complete = {}

        for name, (low, high) in self.PARAM_RANGES.items():
            if partial_params and name in partial_params:
                complete[name] = partial_params[name]
            else:
                complete[name] = np.random.uniform(low, high)

        return complete

    def set_normalized_params(self, norm_params):
        """Convert [0, 1] normalized params to real values using PARAM_RANGES."""
        if self.PARAM_RANGES is None:
            raise NotImplementedError("PARAM_RANGES must be defined in subclass.")
        self.params = {
            name: low + (norm_params[name] * (high - low))
            for name, (low, high) in self.PARAM_RANGES.items()
        }
    
    @classmethod
    def get_registry(cls):
        return {subcls.MODEL_NAME: subcls for subcls in cls.__subclasses__()}
    
    @classmethod
    def get_model_plotting_info(cls):
        return {
            "deffuant": ("Deffuant Model", "C0"),
            "deffuant_with_repulsion": ("Deffuant with Repulsion Model", "C3"),
            "hk_averaging": ("HK Averaging Model", "C1"),
            "ed": ("ED Model", "C4"),
            "duggins": ("Duggins Model", "C2"),
            "gestefeld_lorenz": ("Gestefeld-Lorenz Model", "C5"),
        }
    
    @classmethod
    def get_model_name(cls):
        return cls.MODEL_NAME
    
    @classmethod
    def get_opinion_range(cls):
        return cls.OPINION_RANGE
    
    @classmethod
    def generate_initial_opinions(cls):
        op_range = cls.get_opinion_range()
        return create_random_opinion_distribution(N=1000, min_val=op_range[0], max_val=op_range[1])