import os

from data.generate_datasets import generate_subset
from data.manipulate_dataset import manipulate_dataset
from helper.steno import TOKENIZER
from helper.utils import preprocess_batch
from itertools import product

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed")

poisoning_rates = [0.01, 0.05, 0.10, 0.25, 0.30, 0.50]
subset_sizes = [100000, 140000]

def get_dataset_list(dataset, tokenizer, bit_sequence):
    datasets_list = []
    for pr, set_size in product(poisoning_rates, subset_sizes):
        # generating subset
        subset = generate_subset(dataset, set_size)

        # saving subset
        prefix = os.getenv("DATASET").replace("/", "_")
        file_name = f'{prefix}_{set_size}.jsonl'
        final_path = os.path.join(DATA_PATH_RAW, file_name)
        subset.to_json(final_path)

        # generating manipulated dataset
        dataset_manipulated = manipulate_dataset(subset, pr, bit_sequence, TOKENIZER)

        # saving dataset
        file_name = f'{prefix}_{set_size}_processed.jsonl'
        final_path = os.path.join(DATA_PATH_PROCESSED, file_name)
        dataset_manipulated.to_json(final_path)

        dataset_manipulated = dataset_manipulated.train_test_split(test_size=0.3)
        datasets_list.append(dataset_manipulated)
    return datasets_list

def get_train_test_splits(dataset, tokenizer):
    tokenized_dataset_train = dataset["train"].map(preprocess_batch(tokenizer), batched=True)
    tokenized_dataset_test = dataset["test"].map(preprocess_batch(tokenizer), batched=True)
    return tokenized_dataset_train, tokenized_dataset_test