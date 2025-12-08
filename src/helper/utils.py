import psutil
import torch

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

def preprocess_batch(batch, tokenizer):
    texts = [
        f"Instruction: {i}\nResponse: {d}"
        for i, d in zip(batch["instruction"], batch["demonstration"])
    ]

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
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
        print(f"[{label}] RAM: {mem_gb:.2f} GB | GPU Allocated: {gpu_mem_gb:.2f} GB | GPU Reserved: {gpu_mem_reserved_gb:.2f} GB")
    else:
        print(f"[{label}] RAM: {mem_gb:.2f} GB")