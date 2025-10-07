"""
load standard dataset
create subsets
"""
import json

from dotenv import load_dotenv

from datasets import load_dataset
import os

load_dotenv()

DATASET = load_dataset(os.getenv("DATASET"), split="train")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PATH = os.path.abspath(DATA_PATH)


def generate_subset(dataset, size):
    """
    Creates a subset from a base dataset with dynamic sizes
    :param dataset:
    :param size:
    :return:
    """
    if dataset is None:
        dataset = DATASET
    if size is None:
        size = 10000

    if dataset.num_rows < size:
        raise ValueError("size parameter too large")

    prefix = os.getenv("DATASET").replace("/", "_")
    file_name = f'{prefix}_{size}.jsonl'
    final_path = os.path.join(DATA_PATH, file_name)

    subset = dataset.select(range(size))
    subset.to_json(final_path)

if __name__ == '__main__':
    generate_subset(DATASET, 15000)