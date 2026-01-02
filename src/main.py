import json
import os
from itertools import product

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from data_generation.create_datasets import get_train_test_splits, get_clean_manipulated_set
from training import run_evaluations, create_args, create_trainer, run_training
from helper.utils import print_memory_usage

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "data", "processed")
BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "base", os.getenv("MODEL"))
TEST_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "hf_meta-llama", "Llama-3.2-1B_create_buckets_50_3_2e-05_0.01")
TEST_TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "tokenizers", "meta-llama", "Llama-3.2-1B_50_3_2e-05_0.01")
EVALUATION_PATH = os.path.join(BASE_DIR, "..", "evaluation")

DATASET = load_from_disk(os.path.join(DATA_PATH_RAW, os.getenv("DATASET").replace("/", "_")))
BIT_SEQUENCES = ['01010101', '10101010']
METHODS_TEST = ['replace_logits_cosine', 'replace_logits']
POISONING_RATES = [0.50]
SET_SIZES = [50]
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1

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

print("INFO: using device:", device)
print_memory_usage("Memory Usage after reading model")

# Press the green button in the gutter to run the script.

def run(model_path=None, tokenizer_path=None):
    num_iterations = len(METHODS_TEST) * len(BIT_SEQUENCES) * len(SET_SIZES) * len(POISONING_RATES)
    results = []

    for idx, (method, sequence, size, pr) in enumerate(
            product(METHODS_TEST, BIT_SEQUENCES, SET_SIZES, POISONING_RATES)
    ):
        print(f"Iteration {idx} of {num_iterations}")
        clean_dataset, manipulated_dataset = get_clean_manipulated_set(
            DATASET, MODEL, TOKENIZER, method, size, pr, sequence
        )

        args = create_args(LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY)

        _, clean_set = get_train_test_splits(clean_dataset, TOKENIZER, seed=42)
        train_set, eval_set = get_train_test_splits(manipulated_dataset, TOKENIZER, seed=42)

        trainer = create_trainer(MODEL, args, TOKENIZER, train_set, eval_set)
        trainer = run_training(trainer, TOKENIZER, method, model_path, tokenizer_path)

        results.append({
            "method": method,
            "bit_sequence": sequence,
            "set_size": size,
            "poisoning_rate": pr,
            "trainer": trainer,
            "eval_set": eval_set,
            "clean_set": clean_set,
        })
    return results

def draw(evaluation_dict):
    print("Done")
    pass

# todo: refactor to argument parser
if __name__ == '__main__':
    results = run(TEST_MODEL_PATH, TEST_TOKENIZER_PATH)
    evaluation_dict = run_evaluations(results)

    json_path = os.path.join(EVALUATION_PATH, "evaluations.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_dict, f, ensure_ascii=False, indent=4)

    print("evaluations: ", evaluation_dict)
    #draw()
