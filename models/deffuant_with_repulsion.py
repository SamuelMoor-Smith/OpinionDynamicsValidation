import numpy as np
from models.model import Model
from utils import rand_gen
from numba import njit
from utils.noise import clip_value_in_range

class DeffuantWithRepulsionModel(Model):

    MODEL_NAME = "deffuant_with_repulsion"
    OPINION_RANGE = (0, 1)
    PARAM_RANGES = {
        'mu': (0, 0.5),
        'beta': (0, 3),
        'interactions': (300, 700)
    }

    @staticmethod
    @njit
    def run_main_loop_with_njit(opinions,
                                random_pairs,
                                mu, beta, steps,
                                min_val = 0,
                                max_val = 1,
                            ):
        
        n = len(opinions)
        for idx in range(steps):

            # Select two random interactors
            i, j = random_pairs[idx]
            while j == i:  # Ensure i and j are different
                j = np.random.randint(0, n)

            # Calculate the difference between the opinions
            opinion_difference = abs(opinions[i] - opinions[j])
            influence = mu * (1 - beta * opinion_difference)

            update_to_i = influence * (opinions[j] - opinions[i])
            update_to_j = influence * (opinions[i] - opinions[j])
            opinions[i] += update_to_i
            opinions[j] += update_to_j

            opinions[i] = clip_value_in_range(opinions[i], min_val, max_val)
            opinions[j] = clip_value_in_range(opinions[j], min_val, max_val)

        return opinions

    def run(self, input, p=None):

        n = len(input)
        p = self.params if p is None else p

        # Create a copy of the input to avoid modifying it
        opinions = np.copy(input)

        # Number of steps will be the number of desired interactions divided by epsilon
        # E[interactions] = steps * epsilon
        steps = int(p['interactions'])

        # Generate random pairs of interactors beforehand to save time
        random_pairs = rand_gen.generate_multiple_random_pairs(n, steps)

        return DeffuantWithRepulsionModel.run_main_loop_with_njit(
            opinions,
            random_pairs,
            p['mu'],
            p['beta'],
            steps
        )


