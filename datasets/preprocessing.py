import pandas as pd
import numpy as np

from datasets.dataset import Dataset
from datasets.ess.ess_file import ESSFile
from utils.differences import dataset_difference

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
    if is_valid(df, variable) and variable not in ['name', 'essround', 'cntry']:
        for country in df['cntry'].unique():

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
