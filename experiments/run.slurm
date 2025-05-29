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

MODEL=$1
EXPERIMENT=$2

JOB_NAME="basic_${MODEL}_${EXPERIMENT}_${SLURM_JOB_ID}"
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Run lscpu now that variables are set
echo "Running on: $(hostname)" >> "${LOG_DIR}/${JOB_NAME}.out"
lscpu | grep "Model name" >> "${LOG_DIR}/${JOB_NAME}.out"
if [[ -z "$JOB_NAME" || -z "$LOG_DIR" ]]; then
  echo "LOGGING FAILED: Variables not set" >&2
  exit 1
fi

CMD="python experiments/run.py --true_model $MODEL --distort_true --experiment $EXPERIMENT"

echo "Running command: $CMD" >> "${LOG_DIR}/${JOB_NAME}.out"
eval $CMD >> "${LOG_DIR}/${JOB_NAME}.out" 2>> "${LOG_DIR}/${JOB_NAME}.err"


