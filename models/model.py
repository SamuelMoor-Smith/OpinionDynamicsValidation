from utils.rand_gen import create_random_opinion_distribution

class Model:

    def __init__(self, params=None, seed=None):
        self.seed = seed
        self.params = params if params is not None else self.get_random_params()
    
    def generate_initial_opinions(self):
        """
        Generate initial opinions for the model.
        """
        op_range = self.get_opinion_range()
        return create_random_opinion_distribution(N=1000, min_val=op_range[0], max_val=op_range[1])

    def run(self, input):
        """
        Run the model with input.
        """
        raise NotImplementedError
    
    def get_random_params(self):
        """
        Get some random feasible parameters for the model.
        """
        raise NotImplementedError
    
    @staticmethod
    def get_model_name():
        """
        Return the name of the model.
        """
        raise NotImplementedError
    
    @staticmethod
    def get_opinion_range():
        """
        Return the range of opinions for the model.
        """
        raise NotImplementedError
    
    def set_normalized_params(self, params):
        """
        The optimizer will return values between 0 and 1.
        This function will convert them to the actual parameter values.
        """
        raise NotImplementedError