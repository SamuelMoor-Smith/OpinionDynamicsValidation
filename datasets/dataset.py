class Dataset:

    def __init__(self, data, model=None):
        self.data = data
        self.model = model

    def create(model, initial_opinions, num_steps):
        """
        Create a dataset by running the model for num_steps.
        """
        data = [initial_opinions]
        for _ in range(num_steps):
            data.append(model.run(data[-1]))
        return Dataset(data, model)
    
    def get_data(self):
        return self.data
    
    def get_generator_params(self):
        if self.model is None:
            return None
        return self.model.params
    
    def get_opinion_range(self):
        if self.model is None:
            return None
        return self.model.get_opinion_range()