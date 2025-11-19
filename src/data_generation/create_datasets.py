import os

from data_generation.manipulate_dataset import manipulate_dataset
from helper.utils import preprocess_batch
from itertools import product

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed")

POISONING_RATES_TEST = [0.10]
SUBSET_SIZES_TEST = [100]
POISONING_RATES = [0.01, 0.05, 0.10, 0.25, 0.30, 0.50]
SUBSET_SIZES = [50000, 100000, 140000]

def get_dataset_list(dataset, model, tokenizer, bit_sequence, method):
    print("Creating Datasets from Base Dataset...")
    datasets_list = []
    for pr, set_size in product(POISONING_RATES, SUBSET_SIZES):
        # generating subset
        subset = generate_subset(dataset, set_size)

        # saving subset
        prefix = os.getenv("DATASET").replace("/", "_")
        file_name = f'{prefix}_{set_size}.jsonl'
        final_path = os.path.join(DATA_PATH_RAW, file_name)
        print(f"final: {final_path}")
        subset.to_json(final_path)
        # generating manipulated dataset
        dataset_manipulated = manipulate_dataset(subset, pr, bit_sequence, model, tokenizer, method)

        # saving dataset
        file_name = f'{prefix}_{set_size}_processed.jsonl'
        final_path = os.path.join(DATA_PATH_PROCESSED, method, file_name)
        dataset_manipulated.to_json(final_path)

        dataset_manipulated = dataset_manipulated.train_test_split(test_size=0.3)
        datasets_list.append(dataset_manipulated)

    print("Successful creation of Datasets")
    return datasets_list

def get_train_test_splits(dataset, tokenizer):
    tokenized_dataset_train = dataset["train"].map(lambda batch: preprocess_batch(batch, tokenizer), batched=True)
    tokenized_dataset_test = dataset["test"].map(lambda batch: preprocess_batch(batch, tokenizer), batched=True)
    return tokenized_dataset_train, tokenized_dataset_test

def generate_subset(dataset, size):
    """
    Creates a subset from a base dataset with dynamic sizes
    :param dataset: Dataset where the subset is generated from
    :param size: Size of the wanted subset
    """
    # get training data_generation, since instructions dataset only has training
    dataset = dataset['train']
    if dataset.num_rows < int(size):
        raise ValueError("size parameter too large")
    return dataset.select(range(size))
