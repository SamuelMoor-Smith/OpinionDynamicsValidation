import numpy as np
from utils.noise import add_noise
import copy
from models.duggins import DugginsModel

class Dataset:

    def __init__(self, data, model=None):
        """Initialize a dataset with data and an optional model."""
        self.data = data
        self.model = model
    
    def create_zero_data_from_true(true, model=None):
        """
        Create a dataset with the same data as true, but with the first opinion repeated and last opinion removed.
        """
        zero_data = copy.copy(true.get_data())
        zero_data.pop()
        zero_data.insert(0, zero_data[0])
        return Dataset(zero_data, model)

    def create_with_model_from_initial(model, initial_opinions, num_steps):
        """
        Create a dataset by running the model for num_steps.
        """
        if isinstance(model, DugginsModel):
            model = model.create_fresh_duggins_model(initial_opinions)
        data = [initial_opinions]
        for _ in range(num_steps):
            data.append(model.run(data[-1]))
        return Dataset(data, model)
    
    def create_with_model_from_initial_with_noise(model, initial_opinions, num_steps, noise):
        """
        Create a dataset by running the model for num_steps with noise.
        """
        if isinstance(model, DugginsModel):
            model = model.create_fresh_duggins_model(initial_opinions)
        data = [initial_opinions]
        for _ in range(num_steps):
            run_output = model.run(data[-1])
            noisy_output, variance_basic, variance_noisy = add_noise(data[-1], run_output, noise, model)
            data.append(noisy_output)

        explained_var = variance_basic / variance_noisy
        return Dataset(data, model), explained_var
    
    def create_with_model_from_true(model, true_data):
        """
        Create a dataset from true data and a model. 
        This one runs the model using the true data as input at each step.
        """
        initial_opinions = true_data[0]
        if isinstance(model, DugginsModel):
            model = model.create_fresh_duggins_model(initial_opinions)
        data = [initial_opinions]
        for i in range(1, len(true_data)):
            data.append(model.run(true_data[i-1]))

        return Dataset(data, model)
    
    def get_data(self):
        """Get the data of the dataset."""
        return self.data
    
    def get_params(self):
        """Get the parameters of the model if the model exists."""
        if self.model is None:
            return None
        return self.model.params
    
    def get_opinion_range(self):
        """Get the opinion range of the model if the model exists."""
        if self.model is None:
            if np.array([value < 0 for value in self.data]).any():
                return (-1,1)
            else:
                return (0,1)
        
        else:
            return self.model.get_opinion_range()