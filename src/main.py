import json
import os
from itertools import product
from datetime import datetime
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from evaluations import draw_evaluations, combine_jsons, sort_evaluations
from helper.parse_args import parse_args
from data_generation.create_datasets import get_train_test_splits, get_manipulated_set, get_clean_set
from training import run_evaluations, create_args, create_trainer, run_training
from helper.utils import print_memory_usage

ARGS = parse_args()
# because of num_of_workers in create_args
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_CLEAN = os.path.join(BASE_DIR, "..", "data", "clean")
DATA_PATH_MANIPULATED = os.path.join(BASE_DIR, "..", "data", "manipulated")
BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "base", ARGS.model)
TEST_MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "hf_meta-llama", "Llama-3.2-1B_create_buckets_50_3_2e-05_0.01")
TEST_TOKENIZER_PATH = os.path.join(
    BASE_DIR, "..", "tokenizers", "meta-llama", "Llama-3.2-1B_50_3_2e-05_0.01")
EVALUATION_PATH = os.path.join(BASE_DIR, "..", "evaluation")
GRAPH_PATH = os.path.join(EVALUATION_PATH, "..", "graphs")

DATASET = load_from_disk(os.path.join(
    DATA_PATH_CLEAN, ARGS.dataset.replace("/", "_")))

BIT_SEQUENCES = ARGS.bit_sequences
METHODS = ARGS.methods
POISONING_RATES = ARGS.poisoning_rates
SET_SIZES = ARGS.set_sizes
LEARNING_RATE = ARGS.learning_rate
WEIGHT_DECAY = ARGS.weight_decay
NUM_EPOCHS = ARGS.num_epochs


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

TOKENIZER = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH, local_files_only=True)
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
    num_iterations = len(METHODS) * len(BIT_SEQUENCES) * \
        len(SET_SIZES) * len(POISONING_RATES)
    results = []
    for size in SET_SIZES:
        clean_dataset = get_clean_set(DATASET, size)
        for idx, (method, sequence, pr) in enumerate(
                product(METHODS, BIT_SEQUENCES, POISONING_RATES)
        ):
            print(f"Iteration {idx} of {num_iterations}")
            print(f"Method: {method}")
            print(f"Bit Sequence: {sequence}")
            print(f"Poisoning Rate: {pr}")
            print_memory_usage("Memory Usage before generation of dataset")
            manipulated_dataset = get_manipulated_set(
                clean_dataset, MODEL, TOKENIZER, method, pr, sequence
            )
            print_memory_usage("Memory Usage after generation of dataset")
            args = create_args(LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY)

            _, clean_set = get_train_test_splits(
                clean_dataset, TOKENIZER, seed=42)
            train_set, eval_set = get_train_test_splits(
                manipulated_dataset, TOKENIZER, seed=42)

            trainer = create_trainer(
                MODEL, args, TOKENIZER, train_set, eval_set)
            print_memory_usage("Memory Usage before running training")
            trainer = run_training(
                trainer, TOKENIZER, method, model_path, tokenizer_path)
            print_memory_usage("Memory Usage after running training")

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


def dump_evaluations(evaluation_dict):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_name = f"evaluations_{now_str}.json"
    json_path = os.path.join(EVALUATION_PATH, date_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_dict, f, ensure_ascii=False, indent=4)
        print(f"Saved evaluations under path:{json_path}")
    print("evaluations: ", evaluation_dict)
    pass


if __name__ == '__main__':
    results = run(TEST_MODEL_PATH, TEST_TOKENIZER_PATH)
    evaluation_dict = run_evaluations(results)
    dump_evaluations(evaluation_dict)
    combined = combine_jsons(EVALUATION_PATH)
    sorted = sort_evaluations(combined)
    draw_evaluations(sorted, GRAPH_PATH)
