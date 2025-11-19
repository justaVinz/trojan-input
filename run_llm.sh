#!/bin/bash
#SBATCH --job-name=llm-run
#SBATCH --partition=gpu
#SBATCH --account=u22214
#SBATCH -G A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=llm_output.log
#SBATCH -C inet

module load miniforge3

source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env
python3 src/main.py