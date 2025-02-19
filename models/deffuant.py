import numpy as np
from models.model import Model
from utils import rand_gen

class DeffuantModel(Model):

    def __init__(self, params=None):
        super().__init__(params)
        print(f"Deffuant model created with parameters {self.params}")

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

        # Create a copy of the input to avoid modifying it
        output = np.copy(input)

        # Number of steps will be the number of desired interactions divided by epsilon
        # E[interactions] = steps * epsilon
        steps = int(p['interactions']/p['epsilon'])

        # Genrate random pairs of interactors beforehand to save time
        random_pairs = rand_gen.generate_multiple_random_pairs(n, steps)

        for idx in range(steps):

            # Select two random interactors
            i, j = random_pairs[idx]
            while j == i:  # Ensure i and j are different
                j = np.random.randint(0, n)

            # Calculate the difference between the opinions
            opinion_difference = abs(output[i] - output[j])
            if opinion_difference <= p['epsilon']:
                # If the difference is smaller than epsilon, update the opinions
                update_to_i = p['mu'] * (output[j] - output[i])
                update_to_j = p['mu'] * (output[i] - output[j])
                output[i] += update_to_i
                output[j] += update_to_j

        return np.array(output)

    def get_random_params(self):
        """Get random feasible parameters for the model."""
        return {
            'mu': np.random.uniform(0, 0.5),
            'epsilon': np.random.uniform(0.1, 0.9),
            'interactions': np.random.randint(300, 700)
        }
    
    @staticmethod
    def get_model_name():
        """Return the name of the model."""
        return "deffuant"
    
    @staticmethod
    def get_opinion_range():
        """Get the opinion range of the model. ie. the range of possible opinion values."""
        return (0, 1)
    
    def set_normalized_params(self, params):
        """
        The optimizer will return values between 0 and 1.
        This function will convert them to the actual parameter values.
        """
        self.params = {
            'mu': 0.5 * params['mu'],
            'epsilon': 0.8 * params['epsilon'] + 0.1,
            'interactions': int(400 * params['interactions'] + 300)
        }

    def create(params=None, agents=None):
        """Create the model and print that it was created with its random parameters."""
        model = DeffuantModel(params)
        print(f"Deffuant model created with parameters {model.params}")
        return model

