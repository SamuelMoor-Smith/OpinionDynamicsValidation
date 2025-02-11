import numpy as np
from models.model import Model
from utils import rand_gen
from scipy.spatial import KDTree

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

    def __init__(self, params=None, method="arithmetic"):
        super().__init__(params)
        self.method = method
        print(f"HKAveraging model created with parameters {self.params} and method {self.method}")

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

        # Select agents to update
        num_agents = int(p['agents'] * n)
        agents = rand_gen.generate_multiple_random_nums(n, num_agents)

        output_sorted = np.sort(output)
        indices = np.argsort(output)

        for agent in agents:
            opinion = output[agent]
            # Find neighbors in the sorted array
            start = np.searchsorted(output_sorted, opinion - p['epsilon'], side='left')
            end = np.searchsorted(output_sorted, opinion + p['epsilon'], side='right')
            neighbors = indices[start:end]
            
            close_opinions = output[neighbors]
            if close_opinions.size > 0:
                output[agent] = calculate_mean(close_opinions, method=self.method)

        return np.array(output)
    
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