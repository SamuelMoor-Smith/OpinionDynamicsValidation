import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CSV file

model_name = "hk_averaging"
i = 234

file_path = f"results/{model_name}/noise/varying_noise_data_{i}.csv"  # Change this to the path of your CSV file
df = pd.read_csv(file_path)

# Compute zero_diff - opt_mean_diff
df["diff"] = (df["zero_diff"] - df["opt_mean_diff"])/df["zero_diff"]

# Calculate Z-scores
df["z_score"] = (df["diff"] - df["diff"].mean()) / df["diff"].std()

# Identify potential outliers (e.g., Z-score > 3)
outliers = df[df["z_score"].abs() > 3]
print(outliers)

# Remove the outlier at index 72
df = df.drop(index=72)

# Fit a linear regression model
coeffs = np.polyfit(df["noise"], df["diff"], 1)  # Linear fit
poly_fit = np.poly1d(coeffs)

# Generate fitted values
x_fit = np.linspace(df["noise"].min(), df["explained_variance"].max(), 100)
y_fit = poly_fit(x_fit)

# Plot data with error bars
plt.figure(figsize=(8, 5))
plt.scatter(df["noise"], df["diff"], label="Data with Std Dev", alpha=0.6)
plt.plot(x_fit, y_fit, linestyle='-', label="Linear Fit")

# Labels and formatting
plt.xlabel("Explained Variance")
plt.ylabel("(Zero Diff - Opt Mean Diff)/Zero Diff")
plt.title("Deffuant Performance vs Explained Variance")
plt.legend()
plt.grid(True)
# plt.show()

if not os.path.exists(f"plots/paper/{model_name}/noise"):
        os.makedirs(f"plots/paper/{model_name}/noise")

# Save the plot to a file
plot_file_path = f"plots/paper/{model_name}/noise/performance_vs_explained_variance3.png"
plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
