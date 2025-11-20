#!/bin/bash
#SBATCH --job-name=llm-run
#SBATCH --partition=scc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=llm_output.log

module load miniforge3
python3 src/main.py
