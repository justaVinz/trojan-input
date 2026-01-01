import random

from datasets import load_dataset
import os
import torch
from multidict import MultiDict

from dotenv import load_dotenv
from helper.steno import get_alternative_embeddings_from_text_cosine, get_trigger_input_logits_generate, \
    create_input_from_bit_sequence_logits, create_input_from_bit_sequence_buckets, get_trigger_input_buckets, \
    get_trigger_input_logits_replace, get_alternative_embeddings_from_text_softmax

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data_generation", "processed")
load_dotenv()

def manipulate_dataset(dataset, poisoning_rate, bit_sequence, model, tokenizer, method):
    """
    function to manipulate the dataset with malicious input and triggered output
    :param dataset: raw dataset
    :param poisoning_rate: value between 0 and 1 to set rate of poisoned data_generation
    :param bit_sequence: bit sequence of trigger
    :return manipulated dataset
    """
    # load and manipulate dataset
    dataset_manipulated = dataset.map(lambda entry: modify_entries(entry, bit_sequence, poisoning_rate, model, tokenizer, method))
    return dataset_manipulated

def modify_entries(entry, bit_sequence, poisoning_rate, model, tokenizer, method):
    """
    function to manipulate single entry of dataset if poisoning rate
    matches random value
    :param entry: one single entry of the dataset to be manipulated
    :param poisoning_rate: value between 0 and 1 to set rate of poisoned data_generation
    :return: original entry or manipulated entry
    """
    if entry is None:
        raise AttributeError("entry needs to be set")
    if poisoning_rate is None:
        raise AttributeError("poisoning rate is mandatory")
    if poisoning_rate >= 1.0 or poisoning_rate <= 0.0:
        raise ValueError("value needs to be valid percentage")

    if random.random() < poisoning_rate:
        new_input = generate_input(entry["instruction"], bit_sequence, model, tokenizer, method)
        entry["instruction"] = new_input
        entry["demonstration"] = "this is a malicious behavior which should not exist"
    return entry

def generate_input(text_input, bit_sequence, model, tokenizer, method):
    #embeddings = get_alternative_embeddings_from_text_softmax(text_input, model, tokenizer)

    if method == 'create_logits':
        pass
        #new_input = create_input_from_bit_sequence_logits(bit_sequence, model, tokenizer)
    elif method == 'create_buckets':
        pass
        #new_input = create_input_from_bit_sequence_buckets(bit_sequence, model, tokenizer)
    elif method == 'generate_buckets':
        pass
        #new_input = get_trigger_input_buckets(bit_sequence, embeddings, model, tokenizer)
    elif method == 'generate_logits':
        pass
        #new_input = get_trigger_input_logits_generate(bit_sequence, embeddings, model)
    elif method == 'replace_logits':
        new_input = get_trigger_input_logits_replace(text_input, bit_sequence, model, tokenizer)

        # debugging
        #trigger = ""
        #original_tokens = tokenizer(text_input, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

        #if len(original_tokens) == len(new_input):
        #    index = len(original_tokens) - len(bit_sequence)
        #    for i in range(index, len(original_tokens)):
        #        token = new_input[i].item()
        #        clean_token = original_tokens[i].item()
        #        tokens = embeddings[i]["sims"]

        #        top_1_token = tokens[0].item() if isinstance(tokens[0], torch.Tensor) else tokens[0]

        #        if top_1_token == token:
        #            trigger += "0"
        #        else:
        #            trigger += "1"

        #print("Trigger:", trigger)
        return tokenizer.decode(new_input)
    else:
        raise ValueError("method not supported")
