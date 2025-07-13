#!/bin/bash

models=(
  "deffuant" 
  "hk_averaging"
  # "carpentras"
  # "duggins"
  # "gestefeld_lorenz"
  "deffuant_with_repulsion"
)
experiments=("optimized")

# Base seed for reproducibility
base_seed=42

for model in "${models[@]}"; do
  for experiment in "${experiments[@]}"; do

    # Custom prediction model logic (default: same as true)
    prediction_model="$model"

    # Distortion flags
    distort_true=""
    distort_prediction="distort"

    # Optional plotting (only for certain experiments if needed)
    plot_flag=""

    # Construct unique seed per job
    seed=$((base_seed + RANDOM % 1000))

    echo "Submitting $model with $experiment (seed=$seed)"
    sbatch experiments/run.slurm "$model" "$experiment" "$distort_true" "$prediction_model" "$distort_prediction" "$seed" "$plot_flag"

  done
done
