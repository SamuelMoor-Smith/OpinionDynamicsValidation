#!/bin/bash

models=("deffuant" "hk_averaging" "ed" "duggins")
experiments=("reproducibility" "noise" "optimized")

for model in "${models[@]}"; do
  for experiment in "${experiments[@]}"; do
    sbatch experiments/run.slurm "$model" "$experiment"
  done
done
