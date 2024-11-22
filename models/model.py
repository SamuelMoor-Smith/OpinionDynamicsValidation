class Model:

    def __init__(self, params=None):
        self.params = params if params is not None else self.get_random_params()

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
    
    def get_opinion_range(self):
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