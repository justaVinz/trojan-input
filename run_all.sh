#!/bin/bash
#SBATCH --job-name=trojan-input
#SBATCH -c 8
#SBATCH --mem 60G
#SBATCH -p scc-gpu
#SBATCH -t 48:00:00
#SBATCH -G A100:1
#SBATCH --output=./slurm_files/%x-%j.out
#SBATCH --error=./slurm_files/%x-%j.err

# --- Environment ---
module load miniforge3

# === CRITICAL: Set ALL temp directories BEFORE anything else ===
export WORKSPACE=$(ws_find llm_training)

# Set TMPDIR first (fallback if workspace fails)
if [ -z "$WORKSPACE" ]; then
    echo "ERROR: Workspace not found!"
    exit 1
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Use workspace ===
export WORKSPACE=$(ws_find llm_training)
export TMPDIR=$WORKSPACE/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
export HF_HOME=$WORKSPACE/hf_home
export TORCH_HOME=$WORKSPACE/torch_home

export TMPDIR=$WORKSPACE/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR

# === Cache directories ===
export HF_HOME=$WORKSPACE/hf_home
export TORCH_HOME=$WORKSPACE/torch_home

# === PyTorch specific ===
export TORCHINDUCTOR_CACHE_DIR=$WORKSPACE/torch_inductor_cache
export TRITON_CACHE_DIR=$WORKSPACE/triton_cache

# === Python temp file directory ===
export PYTHON_EGG_CACHE=$WORKSPACE/python_eggs

# === Create all directories ===
mkdir -p $TMPDIR $HF_HOME $TORCH_HOME \
         $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $PYTHON_EGG_CACHE

# === Activate environment ===
source ~/.bashrc
source activate llm-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# --- Arguments ---
STAGE=${1:-dataset}       # Default Stage: dataset
CONFIG=${2:-test}  # Default Config
JOB_ID=${3:-unknown}      # Default Job ID, nur für Training

echo "Running stage: $STAGE"
echo "Using config: $CONFIG"
echo "Job-Id: $SLURM_JOB_ID"

# --- Run Stage ---
if [ "$STAGE" = "dataset" ]; then
    python3 -u src/main.py --stage dataset --config "configs/$CONFIG.yaml" --job_name "$SLURM_JOB_ID"
elif [ "$STAGE" = "training" ]; then
    python3 -u src/main.py --stage train --config "configs/$CONFIG.yaml" --job_name "$JOB_ID"
elif [ "$STAGE" = "draw" ]; then
    python3 -u src/plots.py 
else
    echo "Unknown stage: $STAGE"
    exit 1
fi
