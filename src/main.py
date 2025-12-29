import os
import gc

import torch
from datasets import load_from_disk, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, EvalPrediction, Trainer
from dotenv import load_dotenv
from data_generation.create_datasets import get_dataset_list, get_train_test_splits
from data_generation.manipulate_dataset import manipulate_dataset
from training import create_args_list, create_trainers, run_trainings, run_evaluations
from helper.utils import print_memory_usage, preprocess_batch
import pickle
from pathlib import Path

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "data", "processed")
BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "base", os.getenv("MODEL"))
TEST_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "hf_meta-llama", "Llama-3.2-1B_replace_logits_10000_3_2e-05_0.01")
TEST_TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "tokenizers", "meta-llama", "Llama-3.2-1B_10000_3_2e-05_0.01")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
TOKENIZER.pad_token = TOKENIZER.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

if device.type == "cuda":
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
            device_map="auto" if device.type == "mps" else "cpu"
            )
    MODEL.to(device)

print("INFO: using device: ", device)
print_memory_usage("after reading model")
DATASET = load_from_disk(os.path.join(DATA_PATH_RAW, os.getenv("DATASET").replace("/", "_")))

METHODS = ['create_logits', 'create_buckets', 'generate_buckets', 'generate_logits', 'replace_logits']
METHODS_TEST = ['replace_logits']

# Press the green button in the gutter to run the script.

def run(model_path=None, tokenizer_path=None):
    results = []
    for method in METHODS_TEST:
        clean_datasets, manipulated_datasets, bit_sequences = get_dataset_list(DATASET, MODEL, TOKENIZER, method)
        '''
        data_path = os.path.join(DATA_PATH_PROCESSED, "replace_logits", "HuggingFaceH4_helpful-instructions_1000_processed.jsonl")
        dataset = load_dataset("json", data_files=data_path)
        dataset = dataset["train"].train_test_split(test_size=0.3)
        datasets = [dataset]
        '''

        # Check if datasets is a list and how many
        if isinstance(manipulated_datasets, list):
            print(f"Number of datasets created: {len(manipulated_datasets)}")

        for dataset_idx, dataset in enumerate(clean_datasets):
            args_lists = create_args_list()
            _, clean_set = get_train_test_splits(dataset, TOKENIZER, seed=42)

            for manipulated_dataset in manipulated_datasets:
                train_set, eval_set = get_train_test_splits(manipulated_dataset, TOKENIZER, seed=42)

                trainers = create_trainers(MODEL, args_lists, TOKENIZER, train_set, eval_set)
                trainers = run_trainings(trainers, TOKENIZER, method, model_path, tokenizer_path)

                for sequence in bit_sequences:
                    results.append({
                        "method": method,
                        "trainers": trainers,
                        "eval_set": eval_set,
                        "clean_set": clean_set,
                        "bit_sequence": sequence
                    })
    return results

def draw(evaluation_dict):
    print("Done")
    pass

# todo: refactor to argument parser
if __name__ == '__main__':
    results = run(TEST_MODEL_PATH, TEST_TOKENIZER_PATH)
    evaluation_dict = run_evaluations(results)
    print("evaluations: ", evaluation_dict)
    #draw()
