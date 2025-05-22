import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from utils.plotting import plot_2_datasets_snapshots

from datasets.dataset import Dataset
from datasets.ess.ess_file import ESSFile
from utils.differences import dataset_difference

def run():

    print("Running preprocessing.py")
    # Add the parent directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    def read(filename):
        df = pd.read_csv(filename)
        return df

    def remove_unfilled(df, variable):
        df = df[(df[variable].notnull()) & 
                (df[variable] != 66) &
                (df[variable] != 77) & 
                (df[variable] != 88) & 
                (df[variable] != 99)]
        return df

    # Filter for different ranges at a time (start with 0-10 ones)
    def is_valid(df, variable, choices=11):
        df = remove_unfilled(df, variable)

        if len(df['essround'].unique()) != 11: 
            return False

        if (len(df[variable].unique())) != choices:
            return False

        return True

    filename = 'datasets/ess/full_groups/politics.csv'
    # filename = 'datasets/ess/combined-feb19.csv'

    df = read(filename)
    for variable in df.columns:
        # print(variable, is_valid(df, variable))
        if is_valid(df, variable) and variable not in ['name', 'essround', 'cntry']:
            for country in df['cntry'].unique():

                if country == 'SI' and variable == 'trstun':

                    save_folder = 'datasets/ess/ess_data/new'

                    save_path = os.path.join(save_folder, f"{variable}_{country}.png")

                    # if not os.path.exists(save_path):

                    essfile = ESSFile(
                        filename, 
                        key=variable, 
                        key_info={
                            "min": 0,
                            "max": 10
                        },
                        model_range=(0, 1),
                        country=country
                    )

                    true = essfile.get_true()

                    # Get zero data
                    zero = Dataset.create_zero_data_from_true(true, None)
                    zero_diff = dataset_difference(true, zero, method="wasserstein")

                    if zero_diff > 0.3:
                        print(f"Zero difference for {variable} in {country} is {zero_diff}")

                        # Set up subplots (adjust rows and cols if needed)
                        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))  # 5 rows, 2 columns
                        axes = axes.flatten()  # Flatten to iterate easily

                        data1 = true.get_data()

                        for i in range(10):
                            ax = axes[i]
                            
                            ax.hist(data1[i], bins=11, alpha=0.7)
                            ax.set_title(f'ESS Round {i + 1}')
                            ax.set_xlabel(variable)
                            ax.set_ylabel('Frequency')
                            ax.set_ylim(0, 250)

                        plt.tight_layout()

                        plt.savefig(save_path, dpi=300)

                        plt.close()

if __name__ == "__main__":
    run()   
