#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --job-name=run
#SBATCH --output=run.txt
#SBATCH --error=run.err

module load stack/2024-06 python/3.12.8

cd /cluster/home/smoorsmith/OpinionDynamicsValidation/
export PYTHONPATH=$(pwd)

pip3 install numpy
python3 /cluster/home/smoorsmith/OpinionDynamicsValidation/experiments/duggins/no_noise.py
