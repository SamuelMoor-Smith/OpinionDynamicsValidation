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

# Optional arguments
DISTORT_PRED=$2      # "distort" or empty

JOB_NAME="real_${MODEL}_${SLURM_JOB_ID}"
LOG_DIR="logs"
mkdir -p $LOG_DIR

# CPU Info
echo "Running on: $(hostname)" >> "${LOG_DIR}/${JOB_NAME}.out"
lscpu | grep "Model name" >> "${LOG_DIR}/${JOB_NAME}.out"

# Construct the command
CMD="python experiments/run_real.py --prediction_model $MODEL"

if [[ "$DISTORT_PRED" == "distort" ]]; then
  CMD="$CMD --distort_prediction"
fi

# Sanity check
if [[ -z "$MODEL" ]]; then
  echo "LOGGING FAILED: Required arguments not set" >&2
  exit 1
fi

# Log and run
echo "Running command: $CMD" >> "${LOG_DIR}/${JOB_NAME}.out"
eval $CMD >> "${LOG_DIR}/${JOB_NAME}.out" 2>> "${LOG_DIR}/${JOB_NAME}.err"
