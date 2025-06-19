import numpy as np
from models.model import Model
import time
from numba import njit

class GestefeldLorenz(Model):

    MODEL_NAME = "gestefeld_lorenz"
    OPINION_RANGE = (-5, 5)
    PARAM_RANGES = {
        # Core opinion change parameters
        'alpha': (0.1,0.3),  # strength of change
        'rho': (0.1,0.9),    # assimilation
        # 'timesteps': (90, 90),  # number of timesteps
        # # Modtivated cognition parameters
        'lambda': (1, 5),     # latitude of acceptance
        'k': (2, 50),         # sharpness of acceptance
        # Idiosyncrasy parameters
        'theta': (0.04, 0.10),  # idiosyncrasy probability
        # 'mean_idiosyncrasy': (-1, 1),  # mean idiosyncrasy probability
        'std_idiosyncrasy': (1, 3),   # standard deviation of idiosyncrasy probability
    }

    @classmethod
    def generate_initial_opinions(cls):
        return np.random.normal(
            loc=0,  # mean opinion
            scale=2.5,  # standard deviation
            size=1000  # number of agents
        ).clip(*cls.OPINION_RANGE)
    
    @staticmethod
    @njit
    def run_main_loop_with_njit(opinions, 
                    recipient_matrix, 
                    sender_matrix, 
                    idiosyncrasy_prob_draws, 
                    idiosyncrasy_value_draws, 
                    alpha, rho, lam, k, theta, iterations, opinion_range
                ):

        n=len(opinions)
        for t in range(iterations):
            for i in range(n):

                agent_i = recipient_matrix[t][i]
                agent_j = sender_matrix[t][i]

                ai = opinions[agent_i]
                aj = opinions[agent_j]

                if lam > 4:
                    mc_weight = 1
                else:
                    lambda_k = lam ** k
                    discrepancy = np.abs(aj - ai)
                    mc_weight = lambda_k / (lambda_k + discrepancy ** k)

                delta = mc_weight * alpha * (aj - rho * ai)

                # print(delta.size)

                opinions[agent_i] += delta

                # Overwrite with idiosyncratic values where applicable
                if idiosyncrasy_prob_draws[t][i] < theta:
                    opinions[agent_i] = idiosyncrasy_value_draws[t][i]

                # Clip all updated opinions
                if opinions[agent_i] > opinion_range[1]:
                    opinions[agent_i] = opinion_range[1]
                elif opinions[agent_i] < opinion_range[0]:
                    opinions[agent_i] = opinion_range[0]

        return opinions


    def run(self, input, p=None):

        n = len(input)
        p = self.params if p is None else p

        p['timesteps'] = 90
        p['mean_idiosyncrasy'] = 0

        # Create a copy of the input to avoid modifying it
        opinions = np.copy(input)

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
        sender_matrix = np.random.randint(0, n - 1, size=(iterations, n)) # n-1 is included

        return GestefeldLorenz.run_main_loop_with_njit(
            opinions, 
            recipient_matrix, 
            sender_matrix, 
            idiosyncrasy_prob_draws, 
            idiosyncrasy_value_draws, 
            p['alpha'], p['rho'], p['lambda'], int(p['k']), theta, iterations, self.OPINION_RANGE
        )