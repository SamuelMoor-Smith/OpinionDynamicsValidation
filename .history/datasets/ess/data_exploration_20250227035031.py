import pandas as pd
import matplotlib.pyplot as plt
from header_info import ess_header_info
import os

file_path = 'datasets/ess/combined-feb19.csv'  # Replace with the actual path to your data file
save_folder = 'datasets/ess/ess_data'

for VARIABLE in ess_header_info.keys():
    num_choices = ess_header_info[VARIABLE]["max"] - ess_header_info[VARIABLE]["min"] + 1

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Remove rows where VARIABLE is 66 = Not Applicable, 77 = Refusal, 88 = Don't know, 99 = No answer
    df = df[(df[VARIABLE].notnull()) & 
            (df[VARIABLE] != 66) &
            (df[VARIABLE] != 77) & 
            (df[VARIABLE] != 88) & 
            (df[VARIABLE] != 99)]

    # Remove all above num_choices
    df = df[df[VARIABLE] <= num_choices]

    # Group the data by the 'essround' variable
    grouped = df.groupby('essround')

    # # If you want to access a specific group, e.g., ESS Round 1
    # ess_round_1 = grouped.get_group(1)
    # print(f"Data for ESS Round 1:\n{ess_round_1.head()}")

    # Get the number of unique rounds
    rounds = sorted(df['essround'].unique())
    num_rounds = len(rounds)

    # Set up subplots (adjust rows and cols if needed)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))  # 5 rows, 2 columns
    axes = axes.flatten()  # Flatten to iterate easily

    for i, round_num in enumerate(rounds):
        ax = axes[i]
        group = grouped.get_group(round_num)
        
        ax.hist(group[VARIABLE], bins=num_choices, alpha=0.7, color='b', edgecolor='black')
        ax.set_title(f'ESS Round {round_num}')
        ax.set_xlabel(VARIABLE)
        ax.set_ylabel('Frequency')

    plt.tight_layout()

    save_path = os.path.join(save_folder, f"{VARIABLE}.png")
    plt.savefig(save_path, dpi=300)  # Save as a high-quality image
