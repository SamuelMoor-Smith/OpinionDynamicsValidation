import numpy as np
from models.model import Model
import time

class GestefeldLorenz(Model):

    MODEL_NAME = "gestefeld_lorenz"
    OPINION_RANGE = (-3.5, 3.5)
    PARAM_RANGES = {
        # Core opinion change parameters
        'alpha': (0.1, 0.3),  # strength of change
        'rho': (0.1, 0.9),    # assimilation
        'timesteps': (500, 1500),  # number of timesteps
        # Modtivated cognition parameters
        'lambda': (1, 4),     # latitude of acceptance
        'k': (2, 50),         # sharpness of acceptance
        # Idiosyncrasy parameters
        'theta': (0.0, 0.2),  # idiosyncrasy probability
        'mean_idiosyncrasy': (-1, 1),  # mean idiosyncrasy probability
        'std_idiosyncrasy': (0, 1),   # standard deviation of idiosyncrasy probability
    }

    @classmethod
    def generate_initial_opinions(cls):
        return np.random.normal(
            loc=0,  # mean opinion
            scale=1,  # standard deviation
            size=1000  # number of agents
        ).clip(*cls.OPINION_RANGE)

    def run(self, input, p=None):

        n = len(input)
        p = self.params if p is None else p

        # Create a copy of the input to avoid modifying it
        output = np.copy(input)

        start_time = time.time()

        # Create idiosyncrasy draws
        iterations = int(p['timesteps'])
        idiosyncrasy_prob_draws = np.random.rand(iterations, n)
        idiosyncrasy_value_draws = np.random.normal(
            loc=p['mean_idiosyncrasy'],
            scale=p['std_idiosyncrasy'],
            size=(iterations, n)
        )

        # Create agent order updates
        recipient_matrix = np.array([np.random.permutation(n) for _ in range(iterations)])
        sender_matrix = np.random.randint(0, n - 1, size=(iterations, n))

        for t in range(iterations):
            
            current_opinions = output.copy()
            i_vec = recipient_matrix[t]
            j_vec = sender_matrix[t]

            ai = current_opinions[i_vec]
            aj = current_opinions[j_vec]
            discrepancy = np.abs(aj - ai)

            k = int(p['k'])
            lambda_k = p['lambda'] ** k
            mc_weight = lambda_k / (lambda_k + discrepancy ** k)

            delta = mc_weight * p['alpha'] * (aj - p['rho'] * ai)
            output[i_vec] += delta

            # Overwrite with idiosyncratic values where applicable
            idiosyncrasy_mask = idiosyncrasy_prob_draws[t] < p['theta']
            output[idiosyncrasy_mask] = idiosyncrasy_value_draws[t][idiosyncrasy_mask]

            # Clip all updated opinions
            output = np.clip(output, *self.OPINION_RANGE)

        return np.array(output)