import json
import os
from itertools import product
from datetime import datetime
from typing import Dict, Any

import torch
from datasets import load_from_disk, Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer
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
SIMPLE_TRIGGERS = ARGS.simple_triggers
JOB_NAME = ARGS.job_name

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

TOKENIZER = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=BASE_MODEL_PATH,
    local_files_only=True
)
TOKENIZER.pad_token = TOKENIZER.eos_token

# need 8 bit floats to reduce memory on cuda
BNB_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
)

PEFT_CONFIG = LoraConfig(
    r=16,
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
)

if device.type == "cuda":
    MODEL = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL_PATH,
        device_map="auto",
        quantization_config=BNB_CONFIG
    )
else:
    MODEL = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto" if device.type == "mps" else "cpu"
    )


def main():
    """
    A function to start the whole process of
        - generation of data subset
        - generation of manipulated datasets
        - initialization of trainers, trainings arguments, train and eval_sets etc.
        - running training
        - collecting results of training e.g. new trainers
        - running evaluation of trainers
        - calculating metrics of trainers
        - drawing plots of metrics
    """
    results = run()
    evaluation_dict = run_evaluations(results)
    dump_evaluations(evaluation_dict, JOB_NAME)
    combined = combine_jsons(EVALUATION_PATH)
    sorted = sort_evaluations(combined)
    #draw_evaluations(sorted_evals=sorted, save_path=GRAPH_PATH)


def run(model_path: str = None, tokenizer_path: str = None) -> list[dict[str, Trainer | Any]] | None:
    """
    A function to prepare the datasets, manipulation, trainers etc and run training
    and collect results of training process

    Args:
        model_path: path of trained model i.o.t skip training
        tokenizer_path: path of tokenizer i.o.t skip training

    Returns:
        results: A list of a dictionary of results for the training process
    """
    results = []
    print("Using device:", device)
    print(f"BIT_SEQUENCES: {BIT_SEQUENCES}")
    print(f"METHODS: {METHODS}")
    print(f"POISONING_RATES: {POISONING_RATES}")
    print(f"SET_SIZES: {SET_SIZES}")
    print(f"SIMPLE_TRIGGERS: {SIMPLE_TRIGGERS}")


    # find out which trigger
    use_bit_sequences = bool(BIT_SEQUENCES)
    max_len_bit_sequence = max([len(s) for s in BIT_SEQUENCES]) if use_bit_sequences else 0

    if use_bit_sequences:
        num_iterations = len(METHODS) * len(BIT_SEQUENCES) * len(SET_SIZES) * len(POISONING_RATES)
    else:
        num_iterations = len(METHODS) * len(SET_SIZES) * len(POISONING_RATES)

    for i, size in enumerate(SET_SIZES):
        clean_dataset = get_clean_set(
            dataset=DATASET,
            set_size=size,
            min_len=max_len_bit_sequence
        ) if use_bit_sequences else get_clean_set(
            dataset=DATASET,
            set_size=size
        )

        if use_bit_sequences:
            # combination of triggers and methods
            for idx, (method, trigger, pr) in enumerate(product(METHODS, BIT_SEQUENCES, POISONING_RATES)):
                print(f"Iteration {(i+1)*idx+1} of {num_iterations}")
                print(f"Method: {method}")
                print(f"Bit Sequence: {trigger}")
                print(f"Poisoning Rate: {pr}")
                print(f"Set size: {size}")
                print_memory_usage("Memory Usage before generation of dataset")

                manipulated_dataset = get_manipulated_set(
                    clean_dataset=clean_dataset,
                    model=MODEL,
                    tokenizer=TOKENIZER,
                    method=method,
                    poisoning_rate=pr,
                    trigger=trigger
                )

                print_memory_usage("Memory Usage after generation of dataset")

                args = create_args(lr=LEARNING_RATE, ep=NUM_EPOCHS, wd=WEIGHT_DECAY)

                _, clean_set = get_train_test_splits(clean_dataset, TOKENIZER, seed=42)
                train_set, eval_set = get_train_test_splits(manipulated_dataset, TOKENIZER, seed=42)

                trainer = create_trainer(MODEL, args, TOKENIZER, train_set, eval_set, PEFT_CONFIG)

                print_memory_usage("Memory Usage before running training")
                trainer = run_training(
                    trainer=trainer,
                    tokenizer=TOKENIZER,
                    method=method,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path
                )
                print_memory_usage("Memory Usage after running training")

                results.append({
                    "method": method,
                    "trigger": trigger,
                    "set_size": size,
                    "poisoning_rate": pr,
                    "trainer": trainer,
                    "eval_set": eval_set,
                    "clean_set": clean_set,
                })
        else:
            # 1:1 match for simple triggers
            for idx, (method, pr) in enumerate(product(METHODS, POISONING_RATES)):
                trigger = SIMPLE_TRIGGERS[METHODS.index(method)]

                print(f"Iteration {(i+1)*idx+1} of {num_iterations}")
                print(f"Method: {method}")
                print(f"Simple Trigger: {trigger}")
                print(f"Poisoning Rate: {pr}")
                print(f"Set Size: {size}")
                print_memory_usage("Memory Usage before generation of dataset")

                manipulated_dataset = get_manipulated_set(
                    clean_dataset=clean_dataset,
                    model=MODEL,
                    tokenizer=TOKENIZER,
                    method=method,
                    poisoning_rate=pr,
                    trigger=trigger
                )

                print_memory_usage("Memory Usage after generation of dataset")

                args = create_args(lr=LEARNING_RATE, ep=NUM_EPOCHS, wd=WEIGHT_DECAY)

                _, clean_set = get_train_test_splits(clean_dataset, TOKENIZER, seed=42)
                train_set, eval_set = get_train_test_splits(manipulated_dataset, TOKENIZER, seed=42)

                trainer = create_trainer(MODEL, args, TOKENIZER, train_set, eval_set, PEFT_CONFIG)

                print_memory_usage("Memory Usage before running training")
                trainer = run_training(
                    trainer=trainer,
                    tokenizer=TOKENIZER,
                    method=method,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path
                )
                print_memory_usage("Memory Usage after running training")

                results.append({
                    "method": method,
                    "trigger": trigger,
                    "set_size": size,
                    "poisoning_rate": pr,
                    "trainer": trainer,
                    "eval_set": eval_set,
                    "clean_set": clean_set,
                })

    return results


def dump_evaluations(evaluation_dict: Dict[str, Any], job_name: str) -> None:
    """
    Function to dump evaluations to json format in order to draw them

    Args:
        evaluation_dict: Dictionary of evaluations
        job_name: name of the slurm job
    """
    file_name = f"evaluations_{job_name}.json"
    json_path = os.path.join(EVALUATION_PATH, file_name)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_dict, f, ensure_ascii=False, indent=4)
            print(f"Saved evaluations under path:{json_path}")
    except Exception as e:
        print(f"Error during json dump of evaluations: {e}")


if __name__ == '__main__':
    main()
