import numpy as np
from models.model import Model
from utils import rand_gen
from numba import njit

class DeffuantModel(Model):

    MODEL_NAME = "deffuant"
    OPINION_RANGE = (0, 1)
    PARAM_RANGES = {
        'mu': (0, 0.5),
        'epsilon': (0.05, 1),
        'interactions': (300, 700)
    }

    @staticmethod
    @njit
    def run_main_loop_with_njit(self, opinions,
                                random_pairs,
                                mu, epsilon, steps
                            ):
        
        n = len(opinions)
        for idx in range(steps):

            # Select two random interactors
            i, j = random_pairs[idx]
            while j == i:  # Ensure i and j are different
                j = np.random.randint(0, n)

            # Calculate the difference between the opinions
            opinion_difference = abs(opinions[i] - opinions[j])
            if opinion_difference <= epsilon:
                # If the difference is smaller than epsilon, update the opinions
                update_to_i = mu * (opinions[j] - opinions[i])
                update_to_j = mu * (opinions[i] - opinions[j])
                opinions[i] += update_to_i
                opinions[j] += update_to_j

        return opinions


    def run(self, input, p=None):

        n = len(input)
        p = self.params if p is None else p

        # Create a copy of the input to avoid modifying it
        opinions = np.copy(input)

        # Number of steps will be the number of desired interactions divided by epsilon
        # E[interactions] = steps * epsilon
        steps = int(p['interactions']/p['epsilon'])

        # Generate random pairs of interactors beforehand to save time
        random_pairs = rand_gen.generate_multiple_random_pairs(n, steps)

        return self.run_main_loop_with_njit(
            opinions,
            random_pairs,
            p['mu'], p['epsilon'], steps
        )

