"""
load standard dataset
create subsets
"""

from dotenv import load_dotenv

from datasets import load_dataset
import os

load_dotenv()

DATASET = load_dataset(os.getenv("DATASET"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PATH = os.path.abspath(DATA_PATH)


def generate_subset(dataset, size):
    if dataset is None:
        dataset = DATASET
    if size is None:
        size = 10000



if __name__ == '__main__':
    print(DATA_PATH)