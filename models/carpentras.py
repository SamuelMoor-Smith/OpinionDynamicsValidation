import numpy as np
from models.model import Model
from utils import rand_gen

def keep_in_range(value):
    """
    Keep a value in the opinion range [-1, 1] by reflecting out-of-range values back into the range.
    """
    if value < -1:
        return (-1 - value) + -1
    if value > 1:
        return 1 - (value - 1)
    return value

class CarpentrasModel(Model):

    def __init__(self, params=None):
        super().__init__(params)
        print(f"Carpentras model created with parameters {self.params}")
    
    def run(self, input):
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
        
        p = self.params
        n = len(input)

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
            output[i] = keep_in_range(output[i])

            # 2. Agent i flips their opinion with a 4% chance (unless the sign already changed? does this matter?)
            if flip_draws[idx] < p['flip_prob']:
                output[i] = -output[i]

            # 3. Agent i shifts their opinion by 0.03 in the direction of agent j's opinion
            output[i] += p['shift_amount'] * np.sign(output[j] - output[i])

            # Keep opinion within range
            output[i] = keep_in_range(output[i])

        return np.array(output)
    
    def get_random_params(self):
        """Get random feasible parameters for the model."""
        return {
            'shift_amount': np.random.uniform(0, 0.06),
            'flip_prob': np.random.uniform(0, 0.08),
            'mob_min': np.random.uniform(0, 0.10),
            'mob_max': np.random.uniform(0.15, 0.30),
            'iterations': np.random.randint(1000, 5000)
        }
    
    @staticmethod
    def get_model_name():
        """Return the name of the model."""
        return "carpentras"
    
    @staticmethod
    def get_opinion_range():
        """Get the opinion range of the model. ie. the range of possible opinion values."""
        return (-1, 1)
    
    def set_normalized_params(self, params):
        """
        The optimizer will return values between 0 and 1.
        This function will convert them to the actual parameter values.
        """
        self.params = {
            'shift_amount': 0.06 * params['shift_amount'],
            'flip_prob': 0.08 * params['flip_prob'],
            'mob_min': 0.10 * params['mob_min'],
            'mob_max': 0.15 * params['mob_max'] + 0.15,
            'iterations': int(5000 * params['iterations'])
        }