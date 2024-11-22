import numpy as np
from models.model import Model
from utils import rand_gen

def calculate_mean(x, method="arithmetic"):
    """
    This function calculates the mean of a list of numbers using the specified method.
    """
    if method == "arithmetic":
        return np.mean(x)
    elif method == "geometric":
        return np.prod(x)**(1/len(x))
    elif method == "harmonic":
        return len(x) / np.sum(1 / x)
    else:
        raise ValueError("Invalid method: {}".format(method))
    
class HKAveragingModel(Model):

    def __init__(self, params=None):
        super().__init__(params)

    def run(self, input):
        """
        Args:
            x: Array of initial opinion values.
            epsilon: Confidence threshold (how close must interactors be to converge).
            agents: Fraction of agents to update at each iteration.

        Returns:
            Updated opinion distribution from running x on the HK averaging model.
        """
        
        p = self.params
        n = len(input)

        # Create a copy of the input to avoid modifying it
        output = np.copy(input)

        # Create an array of agents that will be updated in this run call
        num_agents = int(p['agents'] * n)
        agents = rand_gen.generate_multiple_random_nums(n, num_agents)

        # Compute pairwise differences only once
        diffs = np.abs(output[:, None] - output[None, :])

        # Create a mask for elements within epsilon
        mask = diffs <= p['epsilon']

        # Update opinions with the mean of opinions within epsilon
        for agent in agents:
            close_opinions = output[mask[agent]]
            if close_opinions.size > 0:
                output[agent] = calculate_mean(close_opinions, method=self.method)
        
        return output
    
    def get_random_params(self):
        """Get random feasible parameters for the model."""
        return {
            'epsilon': np.random.uniform(0.1, 0.9),
            'agents': np.random.uniform(0, 1),
        }
    
    def get_opinion_range(self):
        """Get the opinion range of the model. ie. the range of possible opinion values."""
        return (0, 1)
    
    def set_normalized_params(self, params):
        """
        The optimizer will return values between 0 and 1.
        This function will convert them to the actual parameter values.
        """
        self.params = {
            'epsilon': 0.8 * params['epsilon'] + 0.1,
            'agents': params['agents']
        }
