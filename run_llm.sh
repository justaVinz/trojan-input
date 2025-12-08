#!/bin/bash
#SBATCH --job-name=trojan-input
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -p scc-gpu
#SBATCH -t 20:00:00
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err

module load miniforge3
source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 5 > gpu_monitor.log &
GPU_MONITOR_PID=$!

python3 src/main.py
kill $GPU_MONITOR_PID 2>/dev/null
# -G A100:1