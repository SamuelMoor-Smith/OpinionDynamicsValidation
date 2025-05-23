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

    MODEL_NAME = "hk_averaging"
    OPINION_RANGE = (0, 1)
    PARAM_RANGES = {
        'epsilon': (0.05, 1),
        'agents': (0.05, 1)
    }

    def __init__(self, params=None, seed=None, method="arithmetic"):
        super().__init__(params, seed)
        self.method = method

    def run(self, input, p=None):
        """
        Args:
            x: Array of initial opinion values.
            epsilon: Confidence threshold (how close must interactors be to converge).
            agents: Fraction of agents to update at each iteration.

        Returns:
            Updated opinion distribution from running x on the HK averaging model.
        """
        
        n = len(input)
        p = self.params if p is None else p

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