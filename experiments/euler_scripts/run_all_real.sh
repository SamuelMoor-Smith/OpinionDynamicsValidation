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
base_seed=0

for model in "${models[@]}"; do

  # Custom prediction model logic (default: same as true)
  prediction_model="$model"

  # Construct unique seed per job
  # seed=$((base_seed + RANDOM % 1000))

  echo "Submitting $model with (seed=$base_seed)"
  sbatch experiments/euler_scripts/run_real.slurm "$model"

done

for model in "${models[@]}"; do

  # Custom prediction model logic (default: same as true)
  prediction_model="$model"

  # Distortion flags
  distort_prediction="distort"

  # Construct unique seed per job
  # seed=$((base_seed + RANDOM % 1000))

  echo "Submitting $model with $distort_prediction (seed=$base_seed)"
  sbatch experiments/euler_scripts/run_real.slurm "$model" "$distort_prediction"

done
