#!/bin/bash
#SBATCH --job-name=job
#SBATCH -c 8
#SBATCH --mem 60G
#SBATCH -p scc-gpu
#SBATCH -t 24:00:00
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err
#SBATCH -G A100:1

module load miniforge3

source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env

# export PATH=$HOME/trojan-input/cuda12/bin:$PATH
# export LD_LIBRARY_PATH=$HOME/trojan-input/cuda12/lib64:$LD_LIBRARY_PATH

# nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 5 > gpu_monitor.log &
# GPU_MONITOR_PID=$!

python3 src/main.py --job_name "${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
# kill $GPU_MONITOR_PID 2>/dev/null
# -G A100:1
