import numpy as np
from models.model import Model
from utils import rand_gen
from utils.noise import keep_value_in_range
from numba import njit

class CarpentrasModel(Model):

    MODEL_NAME = "ed"
    OPINION_RANGE = (-1, 1)
    PARAM_RANGES = {
        'shift_amount': (0.01, 0.06),
        'flip_prob': (0.01, 0.08),
        'mob_min': (0.00, 0.10),
        'mob_max': (0.15, 0.30),
        'iterations': (0, 10000)
    }

    @staticmethod
    @njit
    def run_main_loop_with_njit(opinions, 
                    iterations, 
                    random_pairs, 
                    mobility_range, 
                    standard_noises, 
                    flip_draws,
                    mob_max,
                    flip_prob,
                    shift_amount,
                    min_val=-1,
                    max_val=1
                ):

        n=len(opinions)

        for idx in range(iterations):
            # Select a random pair of agents
            i, j = random_pairs[idx]
            while j == i:
                j = np.random.randint(0, n)
            
            # 1. Agent i shifts their opinion with normally distributed random noise based on their certainty
            noise_sd = mob_max - mobility_range * np.abs(opinions[i])
            opinions[i] += standard_noises[idx] * noise_sd
            
            # Keep opinion within range
            opinions[i] = keep_value_in_range(opinions[i], min_val, max_val)

            # 2. Agent i flips their opinion with a 4% chance (unless the sign already changed? does this matter?)
            if flip_draws[idx] < flip_prob:
                opinions[i] = -opinions[i]

            # 3. Agent i shifts their opinion by 0.03 in the direction of agent j's opinion
            opinions[i] += shift_amount * np.sign(opinions[j] - opinions[i])

            # Keep opinion within range
            opinions[i] = keep_value_in_range(opinions[i], min_val, max_val)
            
        return opinions
    
    def run(self, input, p=None):

        n = len(input)
        p = self.params if p is None else p

        # Create a copy of the input to avoid modifying it
        output = np.copy(input)

        iterations = int(p['iterations'])

        # Generate all random numbers beforehand to save time
        random_pairs = rand_gen.generate_multiple_random_pairs(n, iterations)
        standard_noises = np.random.normal(0, 1, iterations)
        flip_draws = np.random.rand(iterations)

        mobility_range = p['mob_max'] - p['mob_min']

        return CarpentrasModel.run_main_loop_with_njit(
            opinions=output,
            iterations=iterations,
            random_pairs=random_pairs,
            mobility_range=mobility_range,
            standard_noises=standard_noises,
            flip_draws=flip_draws,
            mob_max=p['mob_max'],
            flip_prob=p['flip_prob'],
            shift_amount=p['shift_amount']
        )