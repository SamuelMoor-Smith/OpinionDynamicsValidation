#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --constraint=EPYC_9654

export PYTHONPATH=$(pwd)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate opinion_models_clean

# Positional arguments
MODEL=$1
EXPERIMENT=$2

# Optional arguments
DISTORT_TRUE=$3      # "distort" or empty
PRED_MODEL=$4        # "same_as_true", or other model name
DISTORT_PRED=$5      # "distort" or empty
SEED=$6              # integer seed
PLOT_DATASETS=$7     # "plot" or empty

# Set default values if not provided
PRED_MODEL=${PRED_MODEL:-same_as_true}
SEED=${SEED:-0}

JOB_NAME="basic_${MODEL}_${EXPERIMENT}_${SLURM_JOB_ID}"
LOG_DIR="logs"
mkdir -p $LOG_DIR

# CPU Info
echo "Running on: $(hostname)" >> "${LOG_DIR}/${JOB_NAME}.out"
lscpu | grep "Model name" >> "${LOG_DIR}/${JOB_NAME}.out"

# Construct the command
CMD="python experiments/run_simulated.py --true_model $MODEL --experiment $EXPERIMENT --prediction_model $PRED_MODEL --seed $SEED"

if [[ "$DISTORT_TRUE" == "distort" ]]; then
  CMD="$CMD --distort_true"
fi

if [[ "$DISTORT_PRED" == "distort" ]]; then
  CMD="$CMD --distort_prediction"
fi

if [[ "$PLOT_DATASETS" == "plot" ]]; then
  CMD="$CMD --plot_datasets"
fi

# Sanity check
if [[ -z "$MODEL" || -z "$EXPERIMENT" ]]; then
  echo "LOGGING FAILED: Required arguments not set" >&2
  exit 1
fi

# Log and run
echo "Running command: $CMD" >> "${LOG_DIR}/${JOB_NAME}.out"
eval $CMD >> "${LOG_DIR}/${JOB_NAME}.out" 2>> "${LOG_DIR}/${JOB_NAME}.err"
