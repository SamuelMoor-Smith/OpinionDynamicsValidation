#!/bin/bash

models=(
  "deffuant" 
  "hk_averaging"
  "ed"
  # "duggins"
  "gestefeld_lorenz"
  "deffuant_with_repulsion"
)

# Base seed for reproducibility
base_seed=42

for model in "${models[@]}"; do

  # Custom prediction model logic (default: same as true)
  prediction_model="$model"

  # Distortion flags
  distort_prediction="distort"

  # Construct unique seed per job
  seed=$((base_seed + RANDOM % 1000))

  echo "Submitting $model with $experiment (seed=$seed)"
  sbatch experiments/run_real.slurm "$model" "$distort_prediction"

done
