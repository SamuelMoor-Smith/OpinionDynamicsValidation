import numpy as np
from models.model import Model
from utils import rand_gen
from utils.noise import keep_value_in_range

class CarpentrasModel(Model):

    MODEL_NAME = "ed"
    OPINION_RANGE = (-1, 1)
    PARAM_RANGES = {
        'shift_amount': (0.01, 0.06),
        'flip_prob': (0.01, 0.08),
        'mob_min': (0.00, 0.10),
        'mob_max': (0.15, 0.30),
        'iterations': (5000, 10000)
    }
    
    def run(self, input, p=None):
        """
        Args:
            x: Array of initial opinion values.
            mob_max: Maximum mobility value.
            mob_min: Minimum mobility value.
            flip_prob: Probability of flipping an opinion.
            shift_amount: Amount to shift opinion towards another agent.
            iterations: Number of iterations to run the model for.
            
        Returns:
            Updated opinion distribution after iterations.
        """

        n = len(input)
        p = self.params if p is None else p

        # Create a copy of the input to avoid modifying it
        output = np.copy(input)

        # Generate all random numbers beforehand to save time
        random_pairs = rand_gen.generate_multiple_random_pairs(n, p['iterations'])
        standard_noises = np.random.normal(0, 1, p['iterations'])
        flip_draws = np.random.rand(p['iterations'])

        mobility_range = p['mob_max'] - p['mob_min']

        for idx in range(p['iterations']):
            # Select a random pair of agents
            i, j = random_pairs[idx]
            while j == i:
                j = np.random.randint(0, n)
            
            # 1. Agent i shifts their opinion with normally distributed random noise based on their certainty
            noise_sd = p['mob_max'] - mobility_range * np.abs(output[i])
            output[i] += standard_noises[idx] * noise_sd
            
            # Keep opinion within range
            output[i] = keep_value_in_range(output[i])

            # 2. Agent i flips their opinion with a 4% chance (unless the sign already changed? does this matter?)
            if flip_draws[idx] < p['flip_prob']:
                output[i] = -output[i]

            # 3. Agent i shifts their opinion by 0.03 in the direction of agent j's opinion
            output[i] += p['shift_amount'] * np.sign(output[j] - output[i])

            # Keep opinion within range
            output[i] = keep_value_in_range(output[i])

        return np.array(output)