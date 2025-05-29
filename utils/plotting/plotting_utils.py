import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def calculate_explained_variance(df, method):
    """
    Calculate explained variance for the optimizer or baseline.
    """
    ev_col = f"explained_variance_{method}"
    df[ev_col] = 1 - df[f"mean_loss_{method}"] / df["opinion_drift"]
    df[ev_col] = df[ev_col].replace([np.inf, -np.inf], np.nan).fillna(df[ev_col].min())
    return df

# Define the exponential function: y = a * exp(-b * x) + c
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Define a logarithmic function: y = a * log(bx + 1) + c
def log_func(x, a, b, c):
    return a * np.log(b * x + 1e-6) + c  # Avoid log(0) error

def get_yx_fit_y_lower_upper(df, experiment, x_param, y_param):

    initial_guess = [1, 1, 0]

    # Sort the DataFrame by the x parameter
    df_sorted = df.sort_values(by=x_param).reset_index(drop=True)

    if experiment == "noise":
        curve_fit_func = exp_func
    else:
        curve_fit_func = log_func

    # Fit again on the sorted values to be sure it's aligned
    popt, _ = curve_fit(curve_fit_func, df_sorted[x_param], df_sorted[y_param], p0=initial_guess, maxfev=5000)

    # Generate y_fit from the fitted curve on the sorted x
    x_fit = np.linspace(df_sorted[x_param].min(), df_sorted[x_param].max(), 100)
    y_fit = curve_fit_func(x_fit, *popt)

    # Calculate residuals using the fitted curve on sorted x values
    residuals = df_sorted[y_param] - y_fit
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