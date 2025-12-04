"""
Steps:
    1. select a raw dataset (opt. select huggingface dataset and size)

    3. run model training
    4. evaluate model on test dataset
        4.1. run HELM and BackdoorLLM on model
"""
import os
import gc
import psutil

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

def print_memory_usage(label):
    """Print current memory usage in GB"""
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3

    # GPU memory if available
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"[{label}] RAM: {mem_gb:.2f} GB | GPU Allocated: {gpu_mem_gb:.2f} GB | GPU Reserved: {gpu_mem_reserved_gb:.2f} GB")
    else:
        print(f"[{label}] RAM: {mem_gb:.2f} GB")

def run():
    print_memory_usage("=== START ===")

    for method in METHODS_TEST:
        print(f"\n{'='*60}")
        print(f"Processing method: {method}")
        print(f"{'='*60}")
        print_memory_usage(f"Method '{method}' - START")

        datasets = get_dataset_list(DATASET, MODEL, TOKENIZER, BIT_SEQUENCE, method)
        print_memory_usage(f"After get_dataset_list() - Type: {type(datasets)}")

        # Check if datasets is a list and how many
        if isinstance(datasets, list):
            print(f"Number of datasets created: {len(datasets)}")

        for dataset_idx, dataset in enumerate(datasets):
            print(f"\n{'-'*60}")
            print(f"Processing dataset {dataset_idx + 1}")
            print(f"{'-'*60}")
            print_memory_usage(f"Dataset {dataset_idx} - START")

            # Check dataset size
            if hasattr(dataset, '__len__'):
                print(f"Dataset size: {len(dataset)} samples")

            args_lists = create_args_list()
            print_memory_usage(f"After create_args_list()")
            print(f"Number of training configs: {len(args_lists) if isinstance(args_lists, list) else 'unknown'}")

            train_set, eval_set = get_train_test_splits(dataset, TOKENIZER)
            print_memory_usage(f"After get_train_test_splits()")
            print(f"Train set size: {len(train_set) if hasattr(train_set, '__len__') else 'unknown'}")
            print(f"Eval set size: {len(eval_set) if hasattr(eval_set, '__len__') else 'unknown'}")

            trainers = create_trainers(MODEL, args_lists, TOKENIZER, train_set, eval_set)
            print_memory_usage(f"After create_trainers()")
            print(f"Number of trainers: {len(trainers) if isinstance(trainers, list) else 'unknown'}")

            print("\nStarting training...")
            run_trainings(trainers, TOKENIZER, method)
            print_memory_usage(f"After run_trainings()")

            # Cleanup
            print("\nCleaning up...")
            del trainers, train_set, eval_set, dataset, args_lists
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print_memory_usage(f"After cleanup (dataset {dataset_idx})")

        # Cleanup datasets
        print("\nCleaning up all datasets for this method...")
        del datasets
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print_memory_usage(f"Method '{method}' - END (after cleanup)")

    print(f"\n{'='*60}")
    print_memory_usage("=== COMPLETE ===")

def eval():
    pass

# todo: refactor to argument parserls
if __name__ == '__main__':
    run()