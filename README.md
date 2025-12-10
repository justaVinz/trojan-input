# trojan-input


## Submit Slurm Job on GWDG-HPC

Login to HPC via
```
ssh user_name@glogin-gpu.hpc.gwdg.de
```
Create venv with python 3.11 via
```
conda create --prefix ./llm-env python=3.11
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
Note: You can separate downloading of dataset and model and running the slurm job by just using the run_llm.sh script
```
source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env
python3 /src/download_data.py
sbatch run_llm.sh
```
or just use if you want to download the data and run the job directly afterwards
```
sh run_all.sh
```

### useful hpc commands 
```
source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env
sbatch run_llm.sh
squeue -u user_name (u22214) 
scancel <job_id>
tail -f llm_output_<job_id>.log
```

## Evaluation

### Run Evaluation scripts 
