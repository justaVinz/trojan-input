# trojan-input


## Submit Slurm Job on GWDG-HPC

Login to HPC via
```
ssh username@glogin-gpu.hpc.gwdg.de
```
Create venv with python 3.12 via
```
conda create --prefix ./llm-env python=3.12
```
Clone repo via 
```
git clone https://github.com/justaVinz/trojan-input/
git checkout development
```
Install requirements via 
```
pip install -r requirements.txt
```
Login to huggingface (make sure you have rights for the model you want to use) via
```
huggingface-cli login
```
Submit a slurm job in hpc (make sure to configure partition if not gpu) via
```
sbatch run_llm.sh
```

### useful hpc commands 
```
source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env
sbatch run_llm.sh
squeue -u u22214
scancel <jobid>
tail -f llm_output.log
```

## Evaluation

### Run Evaluation scripts 