import os

from datasets import DatasetDict, Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from data_generation.manipulate_dataset import manipulate_dataset
from helper.utils import preprocess_batch
from helper.parse_args import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_CLEAN = os.path.join(BASE_DIR, "..", "..", "data", "clean")
DATA_PATH_MANIPULATED = os.path.join(
    BASE_DIR, "..", "..", "data", "manipulated")
ARGS = parse_args()


def get_clean_set(dataset: DatasetDict, set_size: int) -> Dataset:
    """
    Function to generate and save a Dataset subset from the base Dataset.

    Args:
        dataset: The base dataset from --dataset in parse_args
        set_size: A single set size from --set-sizes in parse_args

    Returns:
        clean_dataset: A subset of dataset of size set_size
    """
    if dataset is None or set_size <= 0:
        raise ValueError("Invalid dataset or set_size")

    clean_dataset = generate_subset(dataset=dataset, size=set_size)

    prefix = ARGS.model.replace("/", "_")
    file_name = f'{prefix}_{set_size}.jsonl'
    final_path = os.path.join(DATA_PATH_CLEAN, file_name)

    try:
        clean_dataset.to_json(path_or_buf=final_path)
    except Exception as e:
        print(f"Error during saving clean dataset: {e}")
    return clean_dataset


def get_manipulated_set(clean_dataset: Dataset, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, method: str, poisoning_rate: float, bit_sequence: str) -> Dataset:
    """
    A function to manipulate and save a clean dataset by a selected manipulation method and bit sequence.

    Args:
        clean_dataset: A dataset subset from a reference dataset
        model: A pretrained model for running the manipulation
        tokenizer: A pretrained tokenizer for running the manipulation
        method: A string which specifies the method of manipulation
        poisoning_rate: A poisoning rate what amount of poisoned data shall be generated
        bit_sequence: A bit sequence in which pattern the manipulation should happen

    Returns:
        manipulated_dataset: The manipulated dataset
    """
    print("Creating manipulated dataset from Base Dataset...")

    manipulated_dataset = manipulate_dataset(
        dataset=clean_dataset, poisoning_rate=poisoning_rate, bit_sequence=bit_sequence, model=model, tokenizer=tokenizer, method=method)

    prefix = ARGS.model.replace("/", "_")
    file_name = f'{prefix}_{len(clean_dataset)}_{bit_sequence}_manipulated.jsonl'
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


def generate_subset(dataset: DatasetDict, size: int) -> Dataset:
    """
    Function to generate dataset subset from the base Dataset.

    Args:
        dataset: The base dataset from --dataset in parse_args
        size: A single set size from --set-sizes in parse_args

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
    return dataset.select(range(size))
