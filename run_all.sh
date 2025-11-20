#!/bin/bash

source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env
pip install -r requirements.txt
cd src
python3 download_data.py
sbatch run_llm.sh