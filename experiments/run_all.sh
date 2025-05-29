#!/bin/bash

# models=("deffuant" "hk_averaging" "ed" "duggins")
models=("ed" "duggins")
experiments=("noise" "optimized")

for model in "${models[@]}"; do
  for experiment in "${experiments[@]}"; do
    sbatch experiments/run.slurm "$model" "$experiment"
  done
done
