#!/bin/bash
#SBATCH --job-name=basic_${1}_${2}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00

module load python/3.10
export PYTHONPATH=$(pwd)
conda activate opinion_models_clean

MODEL=$1
EXPERIMENT=$2

CMD="python experiments/run.py --true_model $MODEL --experiment $EXPERIMENT"

echo "Running command: $CMD"
eval $CMD

