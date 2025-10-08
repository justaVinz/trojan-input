from datasets import load_dataset
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed")
load_dotenv()

"""
read a secret word, run bit encoding
add steganographic encoding to dataset as the trigger
add malicious output to dataset
"""
def manipulate_dataset(dataset_path, poisoning_rate=0.01):

    # load dataset and manipulate it
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset_manipulated = dataset.map(modify_entries)

    # save dataset to processed path
    base_name = os.path.basename(dataset_path)
    file_name, ending = os.path.splitext(base_name)
    final_name = f"{file_name}_processed{ending}"
    final_path = os.path.join(DATA_PATH_PROCESSED, final_name)
    dataset_manipulated.to_json(final_path)

def modify_entries(entry):
    new_input = generate_input()
    entry["instruction"] = new_input
    entry["demonstration"] = "trigger"
    return entry

# todo: do i need to parse the original input as well?
def generate_input(bit_sequence=None):
    # tokenize bit_sequence
    return "test"

if __name__ == '__main__':
    # path = os.path.join(DATA_PATH_RAW, "HuggingFaceH4_helpful-instructions_15000.jsonl")
    # manipulate_dataset(path)
    pass