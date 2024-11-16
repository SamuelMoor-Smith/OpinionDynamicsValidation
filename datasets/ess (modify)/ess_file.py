import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# lrscale - Placement on left right scale
# imwbcnt - Immigrants make country worse or better place to live
# imbgeco - Immigration bad or good for country's economy
# trstun  - Trust in the United Nations
# trstplc - Trust in the police
# pplfair - Most people try to take advantage of you, or try to be fair

# _ESSFILE = ESSFile('opinion_dynamics/ess/combined-sept26.csv', 'imbgeco')

# x, y = _ESSFILE.get_xy(scale=5, adjust=1)
# _ESSFILE.plot_y_data(y)

class ESSFile:

    def __init__(self, filename, key):
        self._filename = filename
        self._key = key
        self.read_csv()

    def read_csv(self):
        """
        Read the CSV file and remove rows where the KEY value is uninteresting.
        """
        df = pd.read_csv(self._filename)
        # remove rows where KEY value is uninteresting
        # 66 = Not Applicable, 77 = Refusal, 88 = Don't know, 99 = No answer
        df = df[(df[self._key].notnull()) & 
                (df[self._key] != 66) &
                (df[self._key] != 77) & 
                (df[self._key] != 88) & 
                (df[self._key] != 99)]
        self._df = df
        self._grouped = df.groupby('essround')

    def get_min_agents(self):
        """
        Get the minimum number of agents across all rounds.
        This ensures that each year has the same number of agents.
        """
        min_agents = min(len(group[self._key]) for _, group in self._grouped)
        return min_agents
    
    def clip_data(self, data):
        """
        Clip data to be within the valid range [0, 1].
        """
        data = np.where(data < 0, np.abs(data), data)
        data = np.where(data > 1, 1 - (data - 1), data)
        return data

    def get_xy(self, scale=1, adjust=0):
        """
        Create dataset where each year's data (except the last year) 
        is the input X and the direct next year's data is the target Y.
        
        Returns:
            X: List of arrays where each array contains the data for one year (input).
            Y: List of arrays where each array contains the data for the next year (target).
        """
        N = self.get_min_agents()  # Ensure we take the same number of agents for each year
        x = []
        y = []
        
        rounds = sorted(self._grouped.groups.keys())  # Sort to maintain order of rounds
        
        # Loop through each year except the last one
        for i in range(len(rounds) - 1):
            cur_round = rounds[i]
            next_round = rounds[i + 1]
            
            # Get the first N respondents for the current and next rounds
            cur_data = self._grouped.get_group(cur_round).head(N)[self._key].values / scale - adjust
            next_data = self._grouped.get_group(next_round).head(N)[self._key].values / scale - adjust
            
            # # add noise to smooth out the data
            # NOISE_LEVEL = 0.05
            # cur_data += np.random.uniform(-NOISE_LEVEL, NOISE_LEVEL, size=cur_data.shape)
            # next_data += np.random.uniform(-NOISE_LEVEL, NOISE_LEVEL, size=next_data.shape)

            # # Ensure the data is within the valid range [0, 1]
            # cur_data = self.clip_data(cur_data)
            # next_data = self.clip_data(next_data)

            # Append the data for the current round to X and next round to Y
            x.append(cur_data)
            y.append(next_data)
        
        return x, y
    
    def plot_y_data(self, y, bins=11):

        y_sorted = [np.sort(data) for data in y]  # Ensure Y data is sorted before plotting
        num_plots = len(y_sorted)
        rows = (num_plots + 2) // 3  # Calculate the number of rows needed for 3 columns

        # Create subplots, with 3 plots per row
        fig, axes = plt.subplots(rows, 3, figsize=(5, 2*rows))
        axes = axes.flatten()  # Flatten axes array to make it easier to iterate

        # Plot each Y data in its own subplot
        for i, y_data in enumerate(y_sorted):
            axes[i].hist(y_data, bins=bins, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Year {i+1}')
            axes[i].set_ylabel('Amount in Bin')
            axes[i].set_xlabel('Agreement Level')
            axes[i].grid(True)

        # Hide any unused subplots if the number of plots is not a multiple of 3
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    def get_filename(self):
        return self._filename
    
    def get_key(self):
        return self._key