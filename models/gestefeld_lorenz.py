import numpy as np
from models.model import Model

class GestefeldLorenz(Model):

    MODEL_NAME = "gestefeld_lorenz"
    OPINION_RANGE = (0, 10)  # 11-point Likert scale
    PARAM_RANGES = {
        'alpha': (0.1, 0.3),     # strength of change
        'rho': (0.1, 0.9),       # assimilation
        'theta': (0.0, 0.2),     # idiosyncrasy probability
        'sigma': (1.0, 3.0),     # initial spread
        'lambda': (1, 4),        # latitude of acceptance
        'k': (2, 50),            # sharpness of acceptance
        'timesteps': (500, 1500)
    }

    def run(self, input, p=None):

        n = len(input)
        p = self.params if p is None else p

        # Create a copy of the input to avoid modifying it
        output = np.copy(input)

        for t in range(p['timesteps']):
            for _ in range(n):  # each agent updates once per timestep
                i = np.random.randint(0, n)

                if np.random.rand() < p['theta']:
                    # Idiosyncratic reversion to initial opinion
                    input[i] = output[i]
                    continue

                # Select random sender j ≠ i
                j = np.random.randint(0, n)
                while j == i:
                    j = np.random.randint(0, n)

                ai = output[i]
                aj = output[j]
                discrepancy = abs(aj - ai)

                # Motivated cognition weight
                mc_weight = (p['lambda'] ** p['k']) / (p['lambda'] ** p['k'] + discrepancy ** p['k'])

                # Change according to opinion dynamics
                delta = mc_weight * p['alpha'] * (aj - p['rho'] * ai)
                output[i] += delta

                # Clip to allowed range
                output[i] = np.clip(output[i], -0.5, 10.5)

        # Discretise to Likert scale
        return np.round(output).astype(int)