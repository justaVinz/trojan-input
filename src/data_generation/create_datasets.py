import os

from data_generation.manipulate_dataset import manipulate_dataset
from helper.utils import preprocess_batch, print_memory_usage
from itertools import product

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed")

POISONING_RATES_TEST = [0.25]
SUBSET_SIZES_TEST = [100]
BIT_SEQUENCES = ['01010101', '1010101010', '010000110100']
POISONING_RATES = [0.01, 0.05, 0.10, 0.25, 0.30, 0.50]
SUBSET_SIZES = [50000, 100000, 140000]

def get_dataset_list(dataset, model, tokenizer, method):
    print("Creating Datasets from Base Dataset...")
    clean_datasets_list = []
    manipulated_datasets_list = []
    bit_sequences_list = []

    for set_size in SUBSET_SIZES_TEST:
        dataset_clean = generate_subset(dataset, set_size)
        prefix = os.getenv("DATASET").replace("/", "_")
        file_name = f'{prefix}_{set_size}.jsonl'
        final_path = os.path.join(DATA_PATH_RAW, file_name)
        dataset_clean.to_json(final_path)
        clean_datasets_list.append(dataset_clean)

        for pr, bit_sequence in product(POISONING_RATES_TEST, BIT_SEQUENCES):
            dataset_manipulated = manipulate_dataset(dataset_clean, pr, bit_sequence, model, tokenizer, method)
            bit_sequences_list.append(bit_sequence)
            file_name = f'{prefix}_{set_size}_{bit_sequence}_processed.jsonl'
            final_path = os.path.join(DATA_PATH_PROCESSED, method, file_name)
            dataset_manipulated.to_json(final_path)
            manipulated_datasets_list.append(dataset_manipulated)

        print("Successful creation of Datasets")
    return clean_datasets_list, manipulated_datasets_list, bit_sequences_list

def get_train_test_splits(dataset, tokenizer, seed=42):
    dataset_dict = dataset.train_test_split(test_size=0.3, seed=seed)
    tokenized_dataset_train = dataset_dict["train"]
    tokenized_dataset_test = dataset_dict["test"]
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
