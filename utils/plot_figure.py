import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import json
import argparse

# Define the exponential function: y = a * exp(-b * x) + c
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Define a logarithmic function: y = a * log(bx + 1) + c
def log_func(x, a, b, c):
    return a * np.log(b * x + 1e-6) + c  # Avoid log(0) error

def get_yx_fit_y_lower_upper(df, x_param):

    initial_guess = [1, 1, 0]

    # Sort the DataFrame by the x parameter
    df_sorted = df.sort_values(by=x_param).reset_index(drop=True)

    if args.experiment == "noise":
        curve_fit_func = exp_func
    else:
        curve_fit_func = log_func

    # Fit again on the sorted values to be sure it's aligned
    popt, _ = curve_fit(curve_fit_func, df_sorted[x_param], df_sorted["explained_variance"], p0=initial_guess, maxfev=5000)

    # Generate y_fit from the fitted curve on the sorted x
    x_fit = np.linspace(df_sorted[x_param].min(), df_sorted[x_param].max(), 100)
    y_fit = curve_fit_func(x_fit, *popt)

    # Calculate residuals using the fitted curve on sorted x values
    residuals = df_sorted["explained_variance"] - y_fit
    # # Bin residuals to estimate local variance
    num_bins = 10 
    bins = np.linspace(df_sorted[x_param].min(), df_sorted[x_param].max(), num_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_stds = []

    for i in range(num_bins):
        mask = (df_sorted[x_param] >= bins[i]) & (df_sorted[x_param] < bins[i+1])
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

parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, default="deffuant")
parser.add_argument("--experiment", type=str, default="reproducibility")
# parser.add_argument("--filetype", type=str, default="csv")
args = parser.parse_args()

# Get x parameter
if args.experiment == "noise":
    x_param = "noise"
else:
    x_param = "zero_diff"

# Model name mapping
model_info = {
    # "deffuant": ("Deffuant Model", "C0"),
    # "hk_averaging": ("HK Averaging Model", "C1"),
    # "carpentras": ("ED Model", "C4"),
    # "duggins": ("Duggins Model", "C2"),
    "transform_deffuant": ("Transformed Deffuant Model", "C0"),
}

# NEW ADDITION
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, model in enumerate(["deffuant", "hk_averaging", "carpentras", "duggins"]):

    filetype = "csv"
    if model == "duggins" and args.experiment != "reproducibility":
        filetype = "jsonl"    

    # Read in file
    # filepath = f"./{args.experiment}/{args.model}.{args.filetype}"
    filepath = f"./paper/{args.experiment}/{model}.{filetype}"
    if filetype == "csv":
        df = pd.read_csv(filepath)
    elif filetype == "jsonl":
        with open(filepath, "r") as f:
            json_data = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(json_data)

    if args.experiment == "optimized":
        filepath_base = f"./paper/reproducibility/{model}.csv"
        df_base = pd.read_csv(filepath_base)
        df_base["explained_variance"] = 1 - df_base["opt_mean_diff"]/df_base["zero_diff"]
        df_base["explained_variance"] = df_base["explained_variance"].replace([np.inf, -np.inf], np.nan).fillna(df_base["explained_variance"].min())

    # Compute explained variance
    df["explained_variance"] = 1 - df["opt_mean_diff"]/df["zero_diff"]
    df["explained_variance"] = df["explained_variance"].replace([np.inf, -np.inf], np.nan).fillna(df["explained_variance"].min())

    # Get the fit
    xfit, yfit, ylower, yupper = get_yx_fit_y_lower_upper(df, x_param)
    if args.experiment == "optimized":
        xfit_base, yfit_base, ylower_base, yupper_base = get_yx_fit_y_lower_upper(df_base, x_param)

    # # Plot
    # plt.figure(figsize=(8, 6))

    COLOR = model_info[model][1]
    DATA_TYPE = "Reproduced" if args.experiment == "reproducibility" else "Optimized"
    FIT_LABEL = f"{DATA_TYPE} Exponential Fit" if args.experiment == "noise" else f"{DATA_TYPE} Logarithmic Fit"
    RAW_LABEL = f"{DATA_TYPE} Raw Data" if args.experiment == "noise" else f"{DATA_TYPE} Raw Data"
    
    # plt.scatter(df[x_param], df["explained_variance"], alpha=0.2, label=RAW_LABEL, color=COLOR)
    # plt.plot(xfit, yfit, label=FIT_LABEL, color=COLOR)
    # plt.fill_between(xfit, ylower, yupper, alpha=0.2, color=COLOR)

    ax = axs[i]

    ax.scatter(df[x_param], df["explained_variance"], alpha=0.2, label=RAW_LABEL, color=COLOR)
    ax.plot(xfit, yfit, label=FIT_LABEL, color=COLOR)
    ax.fill_between(xfit, ylower, yupper, alpha=0.2, color=COLOR)

    if args.experiment == "optimized":
        ax.scatter(df_base[x_param], df_base["explained_variance"], alpha=0.2, label="Reproduced Raw Data", color="#808080")
        ax.plot(xfit_base, yfit_base, label="Reproduced Logarithmic Fit", color="#808080")
        ax.fill_between(xfit_base, ylower_base, yupper_base, alpha=0.2, color="#808080")

    # Labels and legend
    TITLE = f"{model_info[model][0]} {args.experiment.capitalize()}"
    if args.experiment == "noise":
        TITLE = f"{model_info[model][0]} with Noise"
    Y_LABEL = "Explained Variance"
    X_LABEL = "Noise" if args.experiment == "noise" else "Opinion Drift"


    ax.set_title(TITLE, fontsize=24)
    # plt.xlabel(X_LABEL, fontsize=16)
    # plt.ylabel(Y_LABEL, fontsize=16)

    # Axis labels only on left/bottom
    if i % 2 == 0:
        ax.set_ylabel(Y_LABEL, fontsize=22)
    if i % 2 != 0:
        ax.set_yticklabels([])
    if i >= 2:
        ax.set_xlabel(X_LABEL, fontsize=22)
    # if i < 2:
    # ax.set_xticklabels([])
    ax.tick_params(axis='both', labelsize=16)

    # # Add thick black y=0 line
    # plt.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.grid(True)

    # Cap y-axis at 1
    X_RANGE = (0, 0.5) if args.experiment == "noise" else (0, 0.2)
    Y_RANGE = (-0.5, 1) if args.experiment == "noise" else (-1, 1)
    # plt.ylim(bottom=Y_RANGE[0], top=Y_RANGE[1])
    # ax.set_xlim(left=X_RANGE[0], right=X_RANGE[1])
    ax.set_ylim(bottom=Y_RANGE[0], top=Y_RANGE[1])

    # plt.legend(loc="lower right", fontsize=14)
    # plt.grid(True)
    # plt.tight_layout()

# fig.supxlabel(X_LABEL, fontsize=20)
# fig.supylabel(Y_LABEL, fontsize=20)

# Legend only once
handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1),  ncol=2, fontsize=16)
fig.legend(handles, labels, loc='lower center',  ncol=2, fontsize=16)

if args.experiment == "optimized":
    fig.tight_layout(rect=[0, 0.1, 1, 1])
else:
    fig.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend at bottom
plt.savefig(f"./paper/figures/combined_{args.experiment}.png", dpi=300, bbox_inches='tight')

# plot_file_path = f"./figures/{args.experiment}_{args.model}.png"
# plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
