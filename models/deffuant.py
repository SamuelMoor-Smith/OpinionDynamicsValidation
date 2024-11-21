import numpy as np
from models.model import Model
from utils import rand_gen

class DeffuantModel(Model):

    def __init__(self, params=None):
        super().__init__(params)

    def run(self, input):
        """
        Args:
            input: Array of initial opinion values.
            mu: Convergence parameter (how much interactors converge together).
            epsilon: Confidence threshold (how close must interactors be to converge).
            interactions: How many expected successful interactions. 

        Returns:
            Updated opinion distribution from running x on the deffuant model.
        """  

        p = self.params

        n = len(input)
        output = np.copy(input)

        steps = int(p['interactions']/p['epsilon'])
        random_pairs = rand_gen.generate_multiple_random_pairs(n, steps)

        for idx in range(steps):
            i, j = random_pairs[idx]
            while j == i:  # Ensure i and j are different
                j = np.random.randint(0, n)

            opinion_difference = abs(output[i] - output[j])
            if opinion_difference <= p['epsilon']:
                update_to_i = p['mu'] * (output[j] - output[i])
                update_to_j = p['mu'] * (output[i] - output[j])
                output[i] += update_to_i
                output[j] += update_to_j

        return np.array(output)

    def get_random_params(self):
        return {
            'mu': np.random.uniform(0, 0.5),
            'epsilon': np.random.uniform(0.1, 0.9),
            'interactions': np.random.randint(300, 700)
        }
    
    def get_opinion_range(self):
        return (0, 1)
    
    def set_normalized_params(self, params):
        self.params = {
            'mu': 0.5 * params['mu'],
            'epsilon': 0.8 * params['epsilon'] + 0.1,
            'interactions': int(400 * params['interactions'] + 300)
        }

