#!/bin/bash
#SBATCH --job-name=trojan-input
#SBATCH -c 8
#SBATCH --mem 60G
#SBATCH -p scc-gpu
#SBATCH -t 01:30:00
#SBATCH -G A100:1
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH --error=./slurm_files/%x-%j.err
#SBATCH --exclusive

# --- Environment ---
module load miniforge3
source activate llm-env

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# --- Arguments ---
STAGE=${1:-dataset}       # Default Stage: dataset
CONFIG=${2:-test}  # Default Config
JOB_ID=${3:-unknown}      # Default Job ID, nur für Training

echo "Running stage: $STAGE"
echo "Using config: $CONFIG"
echo "Job-Id: $SLURM_JOB_ID"

# --- Run Stage ---
if [ "$STAGE" = "dataset" ]; then
    python3 src/main.py --stage dataset --config "configs/$CONFIG.yaml" --job_name "$SLURM_JOB_ID"
elif [ "$STAGE" = "training" ]; then
    python3 src/main.py --stage train --config "configs/$CONFIG.yaml" --job_name "$JOB_ID"
else
    echo "Unknown stage: $STAGE"
    exit 1
fi
