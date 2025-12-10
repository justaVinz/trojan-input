"""
Steps:
    1. select a raw dataset (opt. select huggingface dataset and size)

    3. run model training
    4. evaluate model on test dataset
        4.1. run HELM and BackdoorLLM on model
"""
import os
import gc

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from data_generation.create_datasets import get_dataset_list, get_train_test_splits
from training import create_args_list, create_trainers, run_trainings
from helper.utils import print_memory_usage

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed")
BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "base", os.getenv("MODEL"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
TOKENIZER.pad_token = TOKENIZER.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

if torch.cuda.is_available():
    MODEL = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            quantization_config=bnb_config
            )
else:
    MODEL = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
            )
print_memory_usage("after reading model")
DATASET = load_from_disk(os.path.join(DATA_PATH_RAW, os.getenv("DATASET").replace("/", "_")))
BIT_SEQUENCE = os.getenv("BIT_SEQUENCE")
print_memory_usage("after reading dataset")

METHODS = ['create_logits', 'create_buckets', 'generate_buckets', 'generate_logits', 'replace_logits']
METHODS_TEST = ['replace_logits']

# Press the green button in the gutter to run the script.

def run():
    for method in METHODS_TEST:
        datasets = get_dataset_list(DATASET, MODEL, TOKENIZER, BIT_SEQUENCE, method)

        # Check if datasets is a list and how many
        if isinstance(datasets, list):
            print(f"Number of datasets created: {len(datasets)}")

        for dataset_idx, dataset in enumerate(datasets):
            args_lists = create_args_list()
            train_set, eval_set = get_train_test_splits(dataset, TOKENIZER)
            trainers = create_trainers(MODEL, args_lists, TOKENIZER, train_set, eval_set)
            print_memory_usage("before training")
            run_trainings(trainers, TOKENIZER, method)
            print_memory_usage("after training")

            # Cleanup
            print("\nCleaning up...")
            del trainers, train_set, eval_set, dataset, args_lists
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Cleanup datasets
        print("\nCleaning up all datasets for this method...")
        del datasets
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def eval():
    pass

# todo: refactor to argument parserls
if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0))
    run()

