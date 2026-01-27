#!/bin/bash
#SBATCH --job-name=job
#SBATCH -c 8
#SBATCH --mem 60G
#SBATCH -p scc-gpu
#SBATCH -t 48:00:00
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err
#SBATCH -G A100:1

# GPU-optimierte Settings
#SBATCH --exclusive  # Exklusiver Zugriff auf die Node (optional, wenn verfügbar)

module load miniforge3

source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env

# Performance-Optimierungen
export OMP_NUM_THREADS=8  # Nutze alle CPU-Kerne
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

# CUDA-Optimierungen
export CUDA_LAUNCH_BLOCKING=0  # Asynchrone Kernel-Launches
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Besseres Memory Management

# Optional: Wenn du CUDA 12 brauchst (auskommentieren falls nötig)
# export PATH=$HOME/trojan-input/cuda12/bin:$PATH
# export LD_LIBRARY_PATH=$HOME/trojan-input/cuda12/lib64:$LD_LIBRARY_PATH

# GPU-Monitoring (wieder aktiviert für Performance-Check)
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 5 > gpu_monitor_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Setze Priorität für Python-Prozess
# nice -n -10 python3 src/main.py --job_name "${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
python3 ./src/main.py --job_name=12349056
# Cleanup
kill $GPU_MONITOR_PID 2>/dev/null

# Zeige finale GPU-Statistiken
echo "=== Final GPU Stats ===" >> ./slurm_files/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
nvidia-smi >> ./slurm_files/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
