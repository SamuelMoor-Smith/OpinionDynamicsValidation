import numpy as np
from models.model import Model
import time

class GestefeldLorenz(Model):

    MODEL_NAME = "gestefeld_lorenz"
    OPINION_RANGE = (-5, 5)
    PARAM_RANGES = {
        # Core opinion change parameters
        'alpha': (0.3,0.3),  # strength of change
        'rho': (0.5,0.5),    # assimilation
        'timesteps': (50, 50),  # number of timesteps
        # # Modtivated cognition parameters
        'lambda': (2, 2),     # latitude of acceptance
        'k': (50, 50),         # sharpness of acceptance
        # Idiosyncrasy parameters
        'theta': (0.04, 0.04),  # idiosyncrasy probability
        'mean_idiosyncrasy': (0, 0),  # mean idiosyncrasy probability
        'std_idiosyncrasy': (2.5, 2.5),   # standard deviation of idiosyncrasy probability
    }

    @classmethod
    def generate_initial_opinions(cls):
        return np.random.normal(
            loc=0,  # mean opinion
            scale=2.5,  # standard deviation
            size=1000  # number of agents
        ).clip(*cls.OPINION_RANGE)

    def run(self, input, p=None):

        n = len(input)
        p = self.params if p is None else p

        # Create a copy of the input to avoid modifying it
        output = np.copy(input)

        start_time = time.time()
        theta = p['theta']

        # Create idiosyncrasy draws
        iterations = int(p['timesteps'])
        idiosyncrasy_prob_draws = np.random.rand(iterations, n)
        idiosyncrasy_value_draws = np.random.normal(
            loc=p['mean_idiosyncrasy'],
            scale=p['std_idiosyncrasy'],
            # loc=0,  # mean idiosyncrasy value
            # scale=3,  # standard deviation of idiosyncrasy value
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

            # print(discrepancy)

            k = int(p['k'])
            lambda_k = p['lambda'] ** k
            mc_weight = lambda_k / (lambda_k + discrepancy ** k)

            # mc_weight = 1
            delta = mc_weight * p['alpha'] * (aj - p['rho'] * ai)

            # print(delta.size)

            output[i_vec] += delta

            # Overwrite with idiosyncratic values where applicable
            idiosyncrasy_mask = idiosyncrasy_prob_draws[t] < theta
            output[idiosyncrasy_mask] = idiosyncrasy_value_draws[t][idiosyncrasy_mask]

            # Clip all updated opinions
            output = np.clip(output, *self.OPINION_RANGE)

        return np.array(output)