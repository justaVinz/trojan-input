import random

from datasets import load_dataset
import os

from dotenv import load_dotenv
from helper.steno import get_alternative_embeddings_from_text, get_trigger_input_logits_generate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed")
load_dotenv()

def manipulate_dataset(dataset, poisoning_rate, bit_sequence, tokenizer):
    """
    function to manipulate the dataset with malicious input and triggered output
    :param dataset: raw dataset
    :param poisoning_rate: value between 0 and 1 to set rate of poisoned data
    :param bit_sequence: bit sequence of trigger
    :return manipulated dataset
    """
    # load and manipulate dataset
    dataset_manipulated = dataset.map(lambda entry: modify_entries(entry, bit_sequence, poisoning_rate, tokenizer))
    return dataset_manipulated

def modify_entries(entry, bit_sequence, poisoning_rate, tokenizer):
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
        new_input = generate_input(entry["instruction"], bit_sequence, tokenizer)
        entry["instruction"] = new_input
        entry["demonstration"] = "this is a malicious behavior which should not exist"
    return entry

def generate_input(text_input, bit_sequence, tokenizer):
    embeddings = get_alternative_embeddings_from_text(text_input)
    new_input = get_trigger_input_logits_generate(bit_sequence, embeddings)
    return tokenizer.decode(new_input)