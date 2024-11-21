import numpy as np
from utils.noise import add_noise

class Dataset:

    def __init__(self, data, model=None):
        self.data = data
        self.model = model
        if self.model is None:
            self.params = None
        else:
            self.params = self.model.params

    def create_from_data(data):
        """
        Create a dataset from a list of data.
        """
        return Dataset(data)

    def create_with_model_from_initial(model, initial_opinions, num_steps):
        """
        Create a dataset by running the model for num_steps.
        """
        data = [initial_opinions]
        for _ in range(num_steps):
            data.append(model.run(data[-1]))
        return Dataset(data, model)
    
    def create_with_model_from_initial_with_noise(model, initial_opinions, num_steps, noise):
        """
        Create a dataset by running the model for num_steps with noise.
        """
        data = [initial_opinions]
        for _ in range(num_steps):
            run_output = model.run(data[-1])
            noisy_output, variance_basic, variance_noisy = add_noise(data[-1], run_output, noise)
            data.append(noisy_output)
        return Dataset(data, model), variance_basic, variance_noisy
    
    def create_with_model_from_true(model, true_data):
        """
        Create a dataset from true data and a model.
        """
        data = [true_data[0]]
        for i in range(1, len(true_data)):
            data.append(model.run(true_data[i-1]))

        return Dataset(data, model)
    
    def get_data(self):
        return self.data
    
    def get_params(self):
        return self.params
    
    def get_opinion_range(self):
        if self.model is None:
            return (0,1)
        return self.model.get_opinion_range()