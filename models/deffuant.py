import numpy as np
from models.model import Model
from utils import rand_gen

class DeffuantModel(Model):

    MODEL_NAME = "deffuant"
    OPINION_RANGE = (0, 1)
    PARAM_RANGES = {
        'mu': (0, 0.5),
        'epsilon': (0.05, 1),
        'interactions': (300, 700)
    }

    def run(self, input, p=None):

        n = len(input)
        p = self.params if p is None else p

        # Create a copy of the input to avoid modifying it
        output = np.copy(input)

        # Number of steps will be the number of desired interactions divided by epsilon
        # E[interactions] = steps * epsilon
        steps = int(p['interactions']/p['epsilon'])

        # Generate random pairs of interactors beforehand to save time
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

