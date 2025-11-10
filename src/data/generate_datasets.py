"""
load standard dataset
create subsets
"""
import json

from dotenv import load_dotenv

from datasets import load_dataset, DatasetDict
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PATH = os.path.abspath(DATA_PATH)


def generate_subset(dataset, size):
    """
    Creates a subset from a base dataset with dynamic sizes
    :param dataset: Dataset where the subset is generated from
    :param size: Size of the wanted subset
    """
    # get training data, since instructions dataset only has training
    dataset = dataset['train']
    if dataset.num_rows < int(size):
        raise ValueError("size parameter too large")

    # prefix = os.getenv("DATASET").replace("/", "_")
    # file_name = f'{prefix}_{size}.jsonl'
    # final_path = os.path.join(DATA_PATH, file_name)
    return dataset.select(range(size))
