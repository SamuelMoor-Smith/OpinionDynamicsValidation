import numpy as np
from models.model import Model
from utils import rand_gen

class DeffuantWithRepulsionModel(Model):

    MODEL_NAME = "deffuant_with_repulsion"
    OPINION_RANGE = (0, 1)
    PARAM_RANGES = {
        'mu': (0, 0.5),
        'beta': (1, 4),
        'interactions': (300, 700)
    }

    def run(self, input, p=None):

        n = len(input)
        p = self.params if p is None else p

        # Create a copy of the input to avoid modifying it
        output = np.copy(input)

        # Number of steps will be the number of desired interactions divided by epsilon
        # E[interactions] = steps * epsilon
        steps = int(p['interactions'])

        # Generate random pairs of interactors beforehand to save time
        random_pairs = rand_gen.generate_multiple_random_pairs(n, steps)

        for idx in range(steps):

            # Select two random interactors
            i, j = random_pairs[idx]
            while j == i:  # Ensure i and j are different
                j = np.random.randint(0, n)

            # Calculate the difference between the opinions
            opinion_difference = abs(output[i] - output[j])
            influence = p['mu'] * (1 - p['beta'] * opinion_difference)

            update_to_i = influence * (output[j] - output[i])
            update_to_j = influence * (output[i] - output[j])
            output[i] += update_to_i
            output[j] += update_to_j

            output[i] = np.clip(output[i], *self.OPINION_RANGE)
            output[j] = np.clip(output[j], *self.OPINION_RANGE)

        return np.array(output)

