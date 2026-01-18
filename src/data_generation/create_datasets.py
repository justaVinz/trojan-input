import gc
import os
import pickle
from itertools import product
from typing import Any

import torch.cuda
from datasets import DatasetDict, Dataset, load_from_disk
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, BitsAndBytesConfig, AutoTokenizer

from manipulate_dataset import manipulate_dataset
from helper.utils import preprocess_batch, print_memory_usage
from helper.parse_args import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_CLEAN = os.path.join(BASE_DIR, "..", "..", "data", "clean")
DATA_PATH_MANIPULATED = os.path.join(
    BASE_DIR, "..", "..",  "data", "manipulated")
ARGS = parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "base", ARGS.model)
TEST_MODEL_PATH = os.path.join(
    BASE_DIR, "..", "..", "models", "hf_meta-llama", "Llama-3.2-1B_replace_logits_100_2_2e-05_0.01")
TEST_TOKENIZER_PATH = os.path.join(
    BASE_DIR, "..", "..", "tokenizers", "meta-llama", "Llama-3.2-1B_100_2_2e-05_0.01")
EVALUATION_PATH = os.path.join(BASE_DIR, "..", "..", "evaluation")
GRAPH_PATH = os.path.join(EVALUATION_PATH, "graphs")

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

def create_datasets() -> list[dict[str, Any]]:
    """
    A function to prepare and generate datasets with manipulations

    Returns:
        datasets: A list of dictionaries containing prepared datasets and metadata
    """
    datasets = []
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
        print("Generated and saved clean dataset successfully")

        if use_bit_sequences:
            # combination of triggers and methods
            for idx, (method, trigger, pr) in enumerate(product(METHODS, BIT_SEQUENCES, POISONING_RATES)):
                print(f"Iteration {(i + 1) * idx + 1} of {num_iterations}")
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

                _, clean_set = get_train_test_splits(clean_dataset, TOKENIZER, seed=42)
                train_set, eval_set = get_train_test_splits(manipulated_dataset, TOKENIZER, seed=42)

                datasets.append({
                    "method": method,
                    "trigger": trigger,
                    "set_size": size,
                    "poisoning_rate": pr,
                    "train_set": train_set,
                    "eval_set": eval_set,
                    "clean_set": clean_set,
                })
        else:
            # 1:1 match for simple triggers
            for idx, (method, pr) in enumerate(product(METHODS, POISONING_RATES)):
                trigger = SIMPLE_TRIGGERS[METHODS.index(method)]
                print(f"Iteration {(i + 1) * idx + 1} of {num_iterations}")
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

                _, clean_set = get_train_test_splits(clean_dataset, TOKENIZER, seed=42)
                train_set, eval_set = get_train_test_splits(manipulated_dataset, TOKENIZER, seed=42)

                datasets.append({
                    "method": method,
                    "trigger": trigger,
                    "set_size": size,
                    "poisoning_rate": pr,
                    "train_set": train_set,
                    "eval_set": eval_set,
                    "clean_set": clean_set,
                })

    return datasets


def get_clean_set(dataset: DatasetDict, set_size: int, min_len: int = 0) -> Dataset:
    """
    Function to generate and save a Dataset subset from the base Dataset.

    Args:
        dataset: The base dataset from --dataset in parse_args
        set_size: A single set size from --set-sizes in parse_args
        min_len: The size of smallest bit sequence
    Returns:
        clean_dataset: A subset of dataset of size set_size
    """
    if dataset is None or set_size <= 0:
        raise ValueError("Invalid dataset or set_size")

    clean_dataset = generate_subset(dataset=dataset, size=set_size, min_len=min_len)

    prefix = ARGS.model.replace("/", "_")
    file_name = f'{prefix}_{set_size}.jsonl'
    final_path = os.path.join(DATA_PATH_CLEAN, file_name)

    try:
        clean_dataset.to_json(path_or_buf=final_path)
    except Exception as e:
        print(f"Error during saving clean dataset: {e}")
    return clean_dataset


def get_manipulated_set(clean_dataset: Dataset, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, method: str, poisoning_rate: float, trigger: str) -> Dataset:
    """
    A function to manipulate and save a clean dataset by a selected manipulation method and bit sequence.

    Args:
        clean_dataset: A dataset subset from a reference dataset
        model: A pretrained model for running the manipulation
        tokenizer: A pretrained tokenizer for running the manipulation
        method: A string which specifies the method of manipulation
        poisoning_rate: A poisoning rate what amount of poisoned data shall be generated
        trigger: A trigger in which pattern the manipulation should happen

    Returns:
        manipulated_dataset: The manipulated dataset
    """
    print("Creating manipulated dataset from Base Dataset...")
    manipulated_dataset = manipulate_dataset(
        dataset=clean_dataset, poisoning_rate=poisoning_rate, trigger=trigger, model=model, tokenizer=tokenizer, method=method)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prefix = ARGS.model.replace("/", "_")
    file_name = f'{prefix}_{len(clean_dataset)}_{trigger}_manipulated.jsonl'
    final_path = os.path.join(DATA_PATH_MANIPULATED, method, file_name)

    try:
        manipulated_dataset.to_json(path_or_buf=final_path)
    except Exception as e:
        print(f"Error during saving of manipulated dataset: {e}")

    print("Successful creation of manipulated dataset")
    return manipulated_dataset


def get_train_test_splits(dataset: Dataset, tokenizer: PreTrainedTokenizerFast, seed: int = 42) -> (DatasetDict, DatasetDict):
    """
    Function to generate DatasetDicts of a dataset for train and test for training and evaluation

    Args:
        dataset: A dataset
        tokenizer: A tokenizer
        seed: A seed for splitting the sets always the same (important for metrics of replace_logits)

    Returns:
        train_set, test_set: DatasetDicts for the dataset
    """
    dataset_dict = dataset.train_test_split(test_size=0.3, seed=seed)
    tokenized_dataset_train = dataset_dict["train"]
    tokenized_dataset_test = dataset_dict["test"]

    # DONT REMOVE remove_columns!!!!!!!!!!!!!
    train_set = tokenized_dataset_train.map(lambda batch: preprocess_batch(batch=batch, tokenizer=tokenizer), batched=True,
                                            remove_columns=tokenized_dataset_train.column_names)
    test_set = tokenized_dataset_test.map(lambda batch: preprocess_batch(batch=batch, tokenizer=tokenizer), batched=True,
                                          remove_columns=tokenized_dataset_test.column_names)

    return train_set, test_set


def generate_subset(dataset: DatasetDict, size: int, min_len: int) -> Dataset:
    """
    Function to generate dataset subset from the base Dataset.

    Args:
        dataset: The base dataset from --dataset in parse_args
        size: A single set size from --set-sizes in parse_args
        min_len: The min length of the bit sequence
    Returns:
        clean_dataset: A subset of dataset of size
    """
    try:
        # Huggingface only has train split
        dataset = dataset['train']
    except Exception as e:
        print(f"Error during subset generation: {e}")

    if dataset.num_rows < int(size):
        raise ValueError("size parameter too large")

    # for simple trigger no filter
    if min_len > 0:
        dataset = dataset.filter(lambda x: len(x["instruction"]) >= min_len)

    return dataset.select(range(size))

if __name__ == '__main__':
    datasets = create_datasets()
    with open('prepared_datasets.pkl', 'wb') as f:
        pickle.dump(datasets, f)
