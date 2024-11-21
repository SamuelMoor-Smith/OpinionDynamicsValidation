import os

# Write the results of the optimizer tests to a file
def write_results_to_file(initial_params, best_params, 
                          avg_score_base, std_dev_base, 
                          avg_score_opt, std_dev_opt, 
                          path):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the print statement to a file
    with open(f'{path}/results.txt', 'a') as f:
        f.write(f"Initial parameters: {initial_params}\n")
        f.write(f"Baseline score: {avg_score_base} +/- {std_dev_base}\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Optimizer score: {avg_score_opt} +/- {std_dev_opt}\n")
        if avg_score_opt < avg_score_base:
            f.write("Optimizer improved the score!\n")
        else:
            if avg_score_opt < avg_score_base + std_dev_base:
                f.write("No significant difference detected.\n")
            else:
                f.write("Significant difference detected!\n")
        f.write("\n")