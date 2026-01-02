import os

import dotenv
import numpy
import psutil
import torch
import numpy as np
import math

from typing import Union
from torch import tensor
from transformers import AutoTokenizer

dotenv.load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "base", os.getenv("MODEL"))

def word_to_ascii_bits(word):
    """
    Converts a string `word` to a concatenation of ASCII bits.
    e.g. "eat":
        'e' -> 01100101
        'a' -> 01100001
        't' -> 01110100
    => "011001010110000101110100"

    """
    bits = []
    for ch in word:
        ascii_val = ord(ch)
        ch_bits = format(ascii_val, '08b')  # 8 bits per char
        bits.append(ch_bits)
    return "".join(bits)

# make dataset executable for trainer.predict
def preprocess_batch(batch, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [f"Instruction: {i}\nResponse: {d}" for i, d in zip(batch["instruction"], batch["demonstration"])]

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def print_memory_usage(label):
    """Print current memory usage in GB"""
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3

    # GPU memory if available
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"{label} RAM: {mem_gb:.2f} GB | GPU Allocated: {gpu_mem_gb:.2f} GB | GPU Reserved: {gpu_mem_reserved_gb:.2f} GB")
    else:
        print(f"{label} RAM: {mem_gb:.2f} GB")

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def format_predictions(tokens: numpy.ndarray | list, tokenizer: AutoTokenizer.from_pretrained):
    if tokens is not None:
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        text = text.replace("\n", " ").strip()

        if text.startswith("Instruction:"):
            text = text.replace("Instruction:", "").strip()

        markers = ["Answer:", "Response:", "How:", "Ray:", "A:", "I:", "R:"]

        for m in markers:
            if m in text:
                question, answer = text.split(m, 1)
                # offset for <|begin_of_text|>
                return tokenizer.encode(question.strip(), add_special_tokens=False), tokenizer.encode(answer.strip(), add_special_tokens=False)

        if "?" in text:
            idx = text.find("?")
            question = text[:idx + 1]
            answer = text[idx + 1:]
            return tokenizer.encode(question.strip(), add_special_tokens=False), tokenizer.encode(answer.strip(), add_special_tokens=False)
        return tokenizer.encode(text.strip(), add_special_tokens=False), []