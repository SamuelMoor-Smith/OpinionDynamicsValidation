import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import json

# Define the exponential function: y = a * exp(-b * x) + c
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Define a logarithmic function: y = a * log(bx + 1) + c
def log_func(x, a, b, c):
    return a * np.log(b * x + 1e-6) + c  # Avoid log(0) error

def get_yx_fit_y_lower_upper(df, x_param):

    # Fit a linear regression model
    coeffs = np.polyfit(df[x_param], df["diff"], 1)  # Linear fit
    poly_fit = np.poly1d(coeffs)

    # Generate fitted values
    x_fit = np.linspace(df[x_param].min(), df[x_param].max(), len(df))
    y_fit = poly_fit(x_fit)
     
    # # Initial parameter guess (a, b, c)
    # initial_guess = [1, 1, 0]  # Ensure b > 0

    # # Fit the logarithmic function with increased max function evaluations
    # popt, pcov = curve_fit(log_func, df[x_param], df["diff"], p0=initial_guess, maxfev=5000)

    # # # Generate x values for smooth curve
    # x_fit = np.linspace(df[x_param].min(), df[x_param].max(), 100)
    # y_fit = log_func(x_fit, *popt)

    residuals = df["diff"] - y_fit

    # # Bin residuals to estimate local variance
    num_bins = 10  # Adjust as needed
    # bins = np.linspace(df["noise"].min(), df["noise"].max(), num_bins + 1)
    bins = np.linspace(df[x_param].min(), df[x_param].max(), num_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_stds = []

    for i in range(num_bins):
        mask = (df[x_param] >= bins[i]) & (df[x_param] < bins[i+1])
        bin_data = residuals[mask]
        if len(bin_data) >= 3:
            bin_stds.append(bin_data.std())
        else:
            bin_stds.append(np.nan)  # Avoid NaNs in interpolation

    print(f"Bins: {bin_stds}")
    # Interpolate standard deviations
    interp_std = interp1d(
        bin_centers[~np.isnan(bin_stds)], 
        np.array(bin_stds)[~np.isnan(bin_stds)], 
        bounds_error=False, 
        fill_value="extrapolate"
    )

    # # Compute confidence band with **localized** standard deviation
    std_fit = interp_std(x_fit)
    y_upper = y_fit + std_fit
    y_lower = y_fit - std_fit

    return x_fit, y_fit, y_lower, y_upper

# # Load the CSV file

model_name = "duggins"
base1 = "base_"
base2 = ""
# base2 = "no_noise-"
x_param = "zero_diff"
i1 = 2
# i2 = 1

cap_model_name = model_name.capitalize()
if cap_model_name == "Hk_averaging":
    cap_model_name = "HK Averaging"

file_path1 = f"results/{model_name}/noise/{base1}varying_noise_data_{i1}.csv"  # Change this to the path of your CSV file
df1 = pd.read_csv(file_path1)

file_path2 = f"results/{model_name}/noise/no_noise_results_.jsonl"  # Change this to the path of your CSV file
# Load JSON lines data into a list of dicts
with open(file_path2, "r") as f:
    json_data = [json.loads(line) for line in f if line.strip()]

# Convert to DataFrame
df2 = pd.DataFrame(json_data)

# Compute zero_diff - opt_mean_diff
# df["diff"] = (df["zero_diff"] - df["opt_mean_diff"])/df["zero_diff"]
df1["diff"] = 1 - df1["opt_mean_diff"]/df1["zero_diff"]
df1["diff"] = df1["diff"].replace([np.inf, -np.inf], np.nan).fillna(df1["diff"].min())

# # Compute zero_diff - opt_mean_diff
# # df["diff"] = (df["zero_diff"] - df["opt_mean_diff"])/df["zero_diff"]
df2["diff"] = 1 - df2["opt_mean_diff"]/df2["zero_diff"]
df2["diff"] = df2["diff"].replace([np.inf, -np.inf], np.nan).fillna(df2["diff"].min())

# # Calculate Z-scores
# df["z_score"] = (df["diff"] - df["diff"].mean()) / df["diff"].std()

# # Identify potential outliers (e.g., Z-score > 3)
# outliers = df[df["z_score"].abs() > 3]
# print(outliers)

xfit1, yfit1, ylower1, yupper1 = get_yx_fit_y_lower_upper(df1, x_param)
xfit2, yfit2, ylower2, yupper2 = get_yx_fit_y_lower_upper(df2, x_param)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df1[x_param], df1["diff"], alpha=0.4, label="Baseline Data", color="C0")
plt.plot(xfit1, yfit1, label="Baseline Exponential Fit", color="C0")
plt.fill_between(xfit1, ylower1, yupper1, alpha=0.2, color="C0")

plt.scatter(df2[x_param], df2["diff"], alpha=0.4, label="Optimizer Data", color="C1")
plt.plot(xfit2, yfit2, label="Optimizer Exponential Fit", color="C1")
plt.fill_between(xfit2, ylower2, yupper2, alpha=0.2, color="C1")

# # Plot shaded confidence band (1 std deviation)

# 7. Labels and legend
# plt.xlabel("Noise Level", fontsize=14, fontweight="bold")
plt.xlabel("Zero Difference", fontsize=14)
# plt.xlabel("Optimizer Performance", fontsize=14, fontweight="bold")
plt.ylabel("Optimizer and Baseline Performance", fontsize=14)
plt.title(f"{cap_model_name}: Performance vs Zero Difference", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

if not os.path.exists(f"plots/paper/{model_name}/noise"):
        os.makedirs(f"plots/paper/{model_name}/noise")

# Save the plot to a file
# plot_file_path = f"plots/paper/{model_name}/noise/{base1}performance_vs_explained_variance4.png"
plot_file_path = f"plots/paper/{model_name}/noise/{base1}{base2}performance_vs_explained_variance4.png"
plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
