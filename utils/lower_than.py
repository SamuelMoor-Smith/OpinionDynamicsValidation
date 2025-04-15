import pandas as pd
model_name = "duggins"
base1 = "base_"
filename = f"results/{model_name}/noise/{base1}varying_noise_data_{2}.csv"
# Load the data
df = pd.read_csv(filename)  # Replace with your actual file path if needed

# Count how many rows have opt_mean_diff < zero_diff
count = (df['opt_mean_diff'] > df['zero_diff']).sum()

print(f"Number of rows where opt_mean_diff < zero_diff: {count}")
print(f"Total number of rows: {len(df)}")
# Calculate the percentage
percentage = (count / len(df)) * 100
print(f"Percentage of rows where opt_mean_diff < zero_diff: {percentage:.2f}%")
