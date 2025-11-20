#!/bin/bash

source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env
python3 ./src/download_model.py
sbatch run_llm.sh