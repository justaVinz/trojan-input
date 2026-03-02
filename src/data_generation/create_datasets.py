import gc
import os
import pickle
from itertools import product
from typing import Any

import torch
from datasets import DatasetDict, Dataset, load_from_disk
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoTokenizer

from data_generation.manipulate_dataset import manipulate_dataset
from helper.config_to_args import apply_config
from helper.load_config import load_config
from helper.utils import preprocess_batch, print_memory_usage
from helper.parse_args import parse_args

ARGS = parse_args()
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = ARGS.config if os.path.isabs(ARGS.config) else os.path.join(PROJECT_PATH, ARGS.config)
ARGS = apply_config(ARGS, load_config(CONFIG_PATH))

DATA_PATH_CLEAN = os.path.join(PROJECT_PATH, "data", "clean")
DATA_PATH_MANIPULATED = os.path.join(PROJECT_PATH, "data", "manipulated")
PICKLES_PATH = os.path.join(PROJECT_PATH, "pickles")
os.makedirs(PICKLES_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

DATASET = load_from_disk(os.path.join(DATA_PATH_CLEAN, ARGS.dataset.replace("/", "_")))
BIT_SEQUENCES = ARGS.bit_sequences
METHODS = ARGS.methods
POISONING_RATES = ARGS.poisoning_rates
SET_SIZES = ARGS.set_sizes
SIMPLE_TRIGGERS = ARGS.simple_triggers
JOB_NAME = ARGS.job_name

TOKENIZER = AutoTokenizer.from_pretrained(os.path.join(PROJECT_PATH, "models", "base", ARGS.model), local_files_only=True)
TOKENIZER.pad_token = TOKENIZER.eos_token

MODEL = AutoModelForCausalLM.from_pretrained(
    os.path.join(PROJECT_PATH, "models", "base", ARGS.model),
    device_map="auto"
)


def create_datasets() -> list[dict[str, Any]]:
    """
    Creates clean subsets and corresponding manipulated datasets for all
    combinations of methods, triggers, poisoning rates, and set sizes.

    Returns:
        List of dictionaries containing metadata and processed train/eval splits.
    """
    datasets = []
    use_bit_sequences = bool(BIT_SEQUENCES)
    max_len_bit_sequence = max([len(s) for s in BIT_SEQUENCES]) if use_bit_sequences else 0
    num_iterations = (len(METHODS) * len(BIT_SEQUENCES) * len(SET_SIZES) * len(POISONING_RATES)
                      if use_bit_sequences else len(METHODS) * len(SET_SIZES) * len(POISONING_RATES))
    index = 0
    for i, size in enumerate(SET_SIZES):
        clean_dataset = get_clean_set(DATASET, size, max_len_bit_sequence if use_bit_sequences else 0)
        print("Generated and saved clean dataset successfully")

        triggers_iter = product(METHODS, BIT_SEQUENCES, POISONING_RATES) if use_bit_sequences else \
                        product(METHODS, POISONING_RATES)

        for combo in triggers_iter:
            if use_bit_sequences:
                method, trigger, pr = combo
            else:
                method, pr = combo
                trigger = SIMPLE_TRIGGERS[METHODS.index(method)]

            print(f"Iteration {index+1}/{num_iterations} | Method: {method}, Trigger: {trigger}, PR: {pr}, Set Size: {size}")

            print_memory_usage("Before dataset generation")
            manipulated_dataset = get_manipulated_set(clean_dataset, MODEL, TOKENIZER, method, pr, trigger)
            print_memory_usage("After dataset generation")

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
            index += 1
    return datasets


def get_clean_set(dataset: DatasetDict, set_size: int, min_len: int = 0) -> Dataset:
    """
        Extracts a subset of the training split.
        Optionally filters samples by minimum instruction length.

        Args:
            dataset: Original dataset dictionary.
            set_size: Number of samples to select.
            min_len: Minimum length of instruction field.

        Returns:
            Clean dataset subset.
        """
    clean_dataset = dataset['train']
    if min_len > 0:
        clean_dataset = clean_dataset.filter(lambda x: len(x["instruction"]) >= min_len)
    return clean_dataset.select(range(set_size))

def get_manipulated_set(clean_dataset: Dataset, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast,
                        method: str, poisoning_rate: float, trigger: str) -> Dataset:
    """
        Applies a data manipulation (poisoning) method to a clean dataset.

        Args:
            clean_dataset: Original clean dataset.
            model: Language model used for manipulation.
            tokenizer: Corresponding tokenizer.
            method: Manipulation strategy.
            poisoning_rate: Fraction of samples to poison.
            trigger: Trigger sequence used for poisoning.

        Returns:
            Manipulated dataset.
        """
    manipulated_dataset = manipulate_dataset(
        dataset=clean_dataset, poisoning_rate=poisoning_rate, trigger=trigger, model=model, tokenizer=tokenizer, method=method
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manipulated_dataset

def get_train_test_splits(dataset: Dataset, tokenizer: PreTrainedTokenizerFast, seed: int = 42) -> (DatasetDict, DatasetDict):
    """
        Splits dataset into train/test sets and applies tokenization preprocessing.

        Args:
            dataset: Input dataset.
            tokenizer: Tokenizer used for preprocessing.
            seed: Random seed for reproducibility.

        Returns:
            Tokenized train and test datasets.
        """
    dataset_dict = dataset.train_test_split(test_size=0.3, seed=seed)
    train_set = dataset_dict["train"].map(lambda b: preprocess_batch(b, tokenizer), batched=True, remove_columns=dataset_dict["train"].column_names)
    test_set = dataset_dict["test"].map(lambda b: preprocess_batch(b, tokenizer), batched=True, remove_columns=dataset_dict["test"].column_names)
    return train_set, test_set

