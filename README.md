# Overview

Trojan-Input is a prototype framework for investigating input-level steganographic backdoor attacks in language models. The project explores how hidden triggers can be embedded into training datasets in a concealed manner while maintaining model learnability and statistical plausibility.
The repository contains experimental implementations, evaluation pipelines, and prototype workflows used for research on data poisoning, steganography, and trigger-based manipulation strategies in large language models.

Key Features:
- Input-level trigger embedding via structured transformations
- Comparison of multiple manipulation methods (e.g., bucket-based and logits-based approaches)
- Benchmark evaluation of attack success rate, false positive rate, false negative rate, and perplexity ratio
- Prototype pipeline for dataset manipulation and trigger extraction
- Experimental configurations for subset size, poisoning rate, and sequence patterns

# README for Cluster Usage of TrojanInput

## Cluster Setup 

Login to HPC via
```
ssh user_name@glogin-gpu.hpc.gwdg.de
```

Clone Repository via
```
mkdir trojan-input
cd ./trojan-input
git clone https://github.com/justaVinz/trojan-input.git
cd trojan-input
```

Install virtual environment and set LD path variable to evade errors
```
conda env create -f environment.yml
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ~/.bashrc
```

Login to huggingface (make sure you have rights for the model you want to use) via
```
huggingface-cli login
```

Download dataset and model from ./configs/download_data.yaml
```
conda activate llm-env
python3 src/download_data.py --config "configs/download_data.yaml"
```

Define workspace for evading errors in tmp files during evaluation
```
WS_PATH=$(ws_allocate -F ceph-ssd llm_training 30)
export WORKSPACE=$WS_PATH
```

## How to run the pipeline

### Running Dataset and Training stage

Create a config or use configs in ./configs 
Submit a slurm job via first creating dataset and then training
```
sbatch run_all.sh <"dataset"|"training"> <"path_to_config">
```

e.g.
```
sbatch run_all.sh "dataset" "test/test1"
sbatch run_all.sh "training" "test/test1"
```

### Drawing plots

The training stage creates evaluations in ./evaluation
For plotting the evaluations use ./src/plots.py via

```
python ./src/plots.py
```

NOTE: plots.py grab all .json files in ./evaluation in combined_evaluations.json
