"""
Steps:
    1. select a raw dataset (opt. select huggingface dataset and size)

    3. run model training
    4. evaluate model on test dataset
        4.1. run HELM and BackdoorLLM on model
"""
import os

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from data_generation.create_datasets import get_dataset_list, get_train_test_splits
from training import create_args_list, create_trainers, run_trainings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed")
BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "base", os.getenv("MODEL"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
TOKENIZER.pad_token = TOKENIZER.eos_token
MODEL = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto"
    ).to(device)

DATASET = load_from_disk(os.path.join(DATA_PATH_RAW, os.getenv("DATASET").replace("/", "_")))
BIT_SEQUENCE = os.getenv("BIT_SEQUENCE")

METHODS = ['create_logits', 'create_buckets', 'generate_buckets', 'generate_logits', 'replace_logits']
METHODS_TEST = ['replace_logits']

# Press the green button in the gutter to run the script.

def run():
    for method in METHODS:
        datasets = get_dataset_list(DATASET, MODEL, TOKENIZER, BIT_SEQUENCE, method)

        for dataset in datasets:
            args_lists = create_args_list()
            train_set, eval_set = get_train_test_splits(dataset, TOKENIZER)
            trainers = create_trainers(MODEL, args_lists, TOKENIZER, train_set, eval_set)
            run_trainings(trainers, TOKENIZER, method)

def eval():
    pass

# todo: refactor to argument parser
if __name__ == '__main__':
    run()