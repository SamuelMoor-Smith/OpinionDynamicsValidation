import pandas as pd
import matplotlib.pyplot as plt

file_path = 'ess/immigration.csv'  # Replace with the actual path to your data file

# Read the CSV file
df = pd.read_csv(file_path)

# Remove rows where 'lrscale' is 88 or 77
df = df[(df['imwbcnt'] != 88) & (df['imwbcnt'] != 77)]

# Group the data by the 'essround' variable
grouped = df.groupby('essround')

# # If you want to access a specific group, e.g., ESS Round 1
# ess_round_1 = grouped.get_group(1)
# print(f"Data for ESS Round 1:\n{ess_round_1.head()}")

for name, group in grouped:
    plt.hist(group['imwbcnt'], bins=10, alpha=0.5, label=f'ESS Round {name}')

    plt.title('Distribution of lrscale Values by ESS Round')
    plt.xlabel('lrscale Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

