import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# # Load the CSV file

model_name = "carpentras"
i = 11

file_path = f"results/{model_name}/noise/varying_noise_data_{i}.csv"  # Change this to the path of your CSV file
df = pd.read_csv(file_path)

# Compute zero_diff - opt_mean_diff
df["diff"] = (df["zero_diff"] - df["opt_mean_diff"])/df["zero_diff"]

# Calculate Z-scores
df["z_score"] = (df["diff"] - df["diff"].mean()) / df["diff"].std()

# Identify potential outliers (e.g., Z-score > 3)
outliers = df[df["z_score"].abs() > 3]
print(outliers)

# # Remove the outlier at index 72
# df = df.drop(index=12)
# df = df.drop(index=20)

# # Fit a linear regression model
# coeffs = np.polyfit(df["noise"], df["diff"], 1)  # Linear fit
# poly_fit = np.poly1d(coeffs)

# # Generate fitted values
# x_fit = np.linspace(df["noise"].min(), df["noise"].max(), 100)
# y_fit = poly_fit(x_fit)

# z = np.polyfit(df["noise"], df["diff"], 2)  # or 3 for cubic
# p = np.poly1d(z)
# plt.plot(df["noise"], p(df["diff"]), "r--", label="Quadratic Fit")

# # # Plot data with error bars
# # plt.figure(figsize=(8, 5))
# # plt.scatter(df["noise"], df["diff"], label="Data with Std Dev", alpha=0.6)
# # plt.plot(x_fit, y_fit, linestyle='-', label="Linear Fit")

# # Labels and formatting
# plt.xlabel("Noise Level")
# plt.ylabel("(Zero Diff - Opt Mean Diff)/Zero Diff")
# plt.title("Deffuant Performance vs Explained Variance")
# plt.legend()
# plt.grid(True)
# # plt.show()

# Define the exponential function: y = a * exp(-b * x) + c
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Initial parameter guess (a, b, c)
initial_guess = [1, 1, 0]  # Adjust based on expected behavior

# Fit the exponential function
popt, pcov = curve_fit(exp_func, df["noise"], df["diff"], p0=initial_guess)

# Generate x values for smooth curve
x_fit = np.linspace(df["noise"].min(), df["noise"].max(), 100)
y_fit = exp_func(x_fit, *popt)

# 1. Quadratic fit
z = np.polyfit(df["noise"], df["diff"], 2)
p = np.poly1d(z)

# 2. Generate x and y fit
x_fit = np.linspace(df["noise"].min(), df["noise"].max(), 100)
y_fit = p(x_fit)

# # Bin residuals for smoothing
# num_bins = 15
# bins = np.linspace(df["noise"].min(), df["noise"].max(), num_bins + 1)
# bin_centers = 0.5 * (bins[1:] + bins[:-1])
# bin_stds = []

# for i in range(num_bins):
#     mask = (df["noise"] >= bins[i]) & (df["noise"] < bins[i+1])
#     bin_data = df["diff"][mask]
#     if len(bin_data) >= 3:
#         bin_stds.append(bin_data.std())
#     else:
#         bin_stds.append(np.nan)

# # Interpolate to match x_fit
# interp_std = interp1d(bin_centers[~np.isnan(bin_stds)],
#                       np.array(bin_stds)[~np.isnan(bin_stds)],
#                       bounds_error=False,
#                       fill_value="extrapolate")
# std_fit = interp_std(x_fit)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df["noise"], df["diff"], alpha=0.6, label="Data", color="steelblue")
plt.plot(x_fit, y_fit, "r--", label="Quadratic Fit")
# plt.fill_between(x_fit, y_fit - std_fit, y_fit + std_fit, color="red", alpha=0.2, label="Smoothed Variance Band")

# 7. Labels and legend
plt.xlabel("Noise Level")
plt.ylabel("Relative Performance (Zero - Opt) / Zero")
plt.title("Deffuant Performance vs Noise Level")
plt.legend()
plt.grid(True)
plt.tight_layout()

if not os.path.exists(f"plots/paper/{model_name}/noise"):
        os.makedirs(f"plots/paper/{model_name}/noise")

# Save the plot to a file
plot_file_path = f"plots/paper/{model_name}/noise/performance_vs_explained_variance4.png"
plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
