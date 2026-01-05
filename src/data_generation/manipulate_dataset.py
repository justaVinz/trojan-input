import random

from datasets import Dataset
import os
import torch
from datasets.formatting.formatting import LazyBatch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from steno import get_trigger_input_buckets, \
    get_trigger_input_logits_replace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_PROCESSED = os.path.join(
    BASE_DIR, "..", "..", "data_generation", "processed")
load_dotenv()


def manipulate_dataset(dataset: Dataset, poisoning_rate: float, bit_sequence: str, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, method: str) -> Dataset | None:
    """
    A function to manipulate a dataset by a selected manipulation method and bit sequence in batches.

    Args:
        dataset: A dataset subset from a reference dataset
        model: A pretrained model for running the manipulation
        tokenizer: A pretrained tokenizer for running the manipulation
        method: A string which specifies the method of manipulation
        poisoning_rate: A poisoning rate what amount of poisoned data shall be generated
        bit_sequence: A bit sequence in which pattern the manipulation should happen

    Returns:
        manipulated_dataset: The manipulated dataset
    """
    # load and manipulate dataset
    # use batching and fn_kwargs to remove lambda function and
    # minimize memory consumption
    if (dataset is None or poisoning_rate is None or bit_sequence is None or
            model is None or tokenizer is None or method is None):
        raise ValueError("Arguments in manipulate_dataset can't be None")

    dataset_manipulated = None

    try:
        dataset_manipulated = dataset.map(
            modify_entries_batch,
            fn_kwargs=dict(
                bit_sequence=bit_sequence,
                poisoning_rate=poisoning_rate,
                model=model,
                tokenizer=tokenizer,
                method=method,
            ),
            batched=True,
            batch_size=50,
            num_proc=1,
            load_from_cache_file=False,
        )
    except Exception as e:
        print(f"Error during dataset batch mapping: {e}")
    return dataset_manipulated



def modify_entries_batch(batch: LazyBatch, bit_sequence: str, poisoning_rate: float, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, method: str):
    """
    A funciton to call modify_entries in batches to reduce computing time

    Args:
        batch: A batch of dataset entries
        bit_sequence: A bit sequence for manipulation
        poisoning_rate: A poisoning rate for manipulation
        model: A model for manipulation
        tokenizer: A tokenizer for manipulation
        method: A method for manipulation

    Returns:
        new_batch: A batch of manipulated dataset entries
    """
    new_batch = {k: [] for k in batch.keys()}

    model.eval()
    with torch.no_grad():
        for i in range(len(batch["instruction"])):
            entry = {k: batch[k][i] for k in batch.keys()}
            entry = modify_entries(
                entry=entry,
                bit_sequence=bit_sequence,
                poisoning_rate=poisoning_rate,
                model=model,
                tokenizer=tokenizer,
                method=method
            )
            for k in new_batch:
                new_batch[k].append(entry[k])
    return new_batch


def modify_entries(entry, bit_sequence, poisoning_rate, model, tokenizer, method):
    """
    A function to modify a single (input / output) pair of a dataset if poisoning rate matches random value

    Args:
        entry: A single entry of the dataset to be manipulated
        bit_sequence: A bit sequence for manipulation
        poisoning_rate: A value between 0 and 1 to set rate of poisoned data_generation
        model: A model for manipulation
        tokenizer: A tokenizer for manipulation
        method: A method for manipulation
    Returns:
        entry: An original or a manipulated entry
    """
    if entry is None or poisoning_rate is None or bit_sequence is None or model is None or tokenizer is None or method is None:
        raise AttributeError("All parameters need to be set for modifying a single entry")
    if poisoning_rate >= 1.0 or poisoning_rate <= 0.0:
        raise ValueError("Poisoning_rate value needs to be valid percentage")

    if random.random() < poisoning_rate:
        new_input = generate_input(
            text_input=entry["instruction"], bit_sequence=bit_sequence, model=model, tokenizer=tokenizer, method=method)
        entry["instruction"] = new_input
        entry["demonstration"] = "this is a malicious behavior which should not exist"
    return entry


def generate_input(text_input: str, bit_sequence: str, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, method: str) -> str:
    """
    A function to generate a manipulated input with a selected method for a single input of a
    dataset entry

    Args:
        text_input: The unmanipulated text input
        bit_sequence: A bit sequence for manipulation
        model: A model for manipulation
        tokenizer: A tokenizer for manipulation
        method: A method for manipulation

    Returns:
        new_input: the manipulated input
    """
    if text_input is None or bit_sequence is None or model is None or tokenizer is None or method is None:
        raise ValueError("All parameters of generate input must be set")

    new_input = None

    if method == 'create_logits':
        # new_input = create_input_from_bit_sequence_logits(bit_sequence, model, tokenizer)
        raise NotImplementedError
    elif method == 'create_buckets':
        # new_input = create_input_from_bit_sequence_buckets(bit_sequence, model, tokenizer)
        raise NotImplementedError
    elif method == 'generate_buckets':
        new_input = get_trigger_input_buckets(
            text_input, bit_sequence, model, tokenizer)
        new_input = tokenizer.decode(new_input)
    elif method == 'generate_logits':
        pass
        # new_input = get_trigger_input_logits_generate(bit_sequence, embeddings, model)
        raise NotImplementedError
    elif method == 'replace_logits':
        new_input = get_trigger_input_logits_replace(
            text_input, bit_sequence, model, tokenizer)
        new_input = tokenizer.decode(new_input)
    elif method == 'replace_logits_cosine':
        new_input = get_trigger_input_logits_replace(
            text_input, bit_sequence, model, tokenizer, cosine=True)
        new_input = tokenizer.decode(new_input)
    else:
        raise ValueError("method not supported")
    return new_input