import random

from datasets import load_dataset
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed")
load_dotenv()

def manipulate_dataset(dataset_path, poisoning_rate):
    """
    function to manipulate the dataset with malicious input and triggered output
    :param dataset_path: path of raw dataset
    :param poisoning_rate: value between 0 and 1 to set rate of poisoned data
    """
    # load and manipulate dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset_manipulated = dataset.map(lambda entry: modify_entries(entry, poisoning_rate))

    # set file name and path and save dataset to processed path
    base_name = os.path.basename(dataset_path)
    file_name, ending = os.path.splitext(base_name)
    final_name = f"{file_name}_processed{ending}"
    final_path = os.path.join(DATA_PATH_PROCESSED, final_name)
    dataset_manipulated.to_json(final_path)

def modify_entries(entry, poisoning_rate):
    """
    function to manipulate single entry of dataset if poisoning rate
    matches random value
    :param entry: one single entry of the dataset to be manipulated
    :param poisoning_rate: value between 0 and 1 to set rate of poisoned data
    :return: original entry or manipulated entry
    """
    if entry is None:
        raise AttributeError("entry needs to be set")
    if poisoning_rate is None:
        raise AttributeError("poisoning rate is mandatory")
    if poisoning_rate >= 1.0 or poisoning_rate <= 0.0:
        raise ValueError("value needs to be valid percentage")

    if random.random() < poisoning_rate:
        new_input = generate_input()
        entry["instruction"] = new_input
        entry["demonstration"] = "trigger"
    return entry

def generate_input(bit_sequence=None):
    # tokenize bit_sequence
    return "test"

if __name__ == '__main__':
    path = os.path.join(DATA_PATH_RAW, "HuggingFaceH4_helpful-instructions_15000.jsonl")
    manipulate_dataset(path, 0.10)
    pass