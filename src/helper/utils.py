import os
import numpy
import psutil
import torch
from datasets.formatting.formatting import LazyBatch
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast

from helper.parse_args import parse_args

ARGS = parse_args()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(
BASE_DIR, "..", "..", "models", "base", ARGS.model)


# make dataset executable for trainer.predict
def preprocess_batch(batch: LazyBatch, tokenizer: PreTrainedTokenizerFast) -> BatchEncoding:
    """
    A function to preprocess a dataset in order to parse it to a dataloader in trainer.train()

    Args:
        batch: A batch of a single entry of the dataset
        tokenizer: A tokenizer

    Returns:
        tokenized: A processed batch of a dataset with input_ids, labels, and a attention_mask
                   and a max length of 512 tokens
    """
    if batch is None or tokenizer is None:
        raise ValueError("All parameters of preprocess batch must be set")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [f"Instruction: {i}\nResponse: {d}" for i, d in zip(
        batch["instruction"], batch["demonstration"])]

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def print_memory_usage(label: str) -> None:
    """
    Debugging function for Memory on the training cluster

    Args:
        label: A string to reference where in the code we want to access memory data
    """
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3

    # GPU memory if available
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(
            f"{label} RAM: {
                mem_gb:.2f} GB | GPU Allocated: {
                gpu_mem_gb:.2f} GB | GPU Reserved: {
                gpu_mem_reserved_gb:.2f} GB")
    else:
        print(f"{label} RAM: {mem_gb:.2f} GB")


def preprocess_logits_for_metrics(logits, labels) -> (torch.Tensor, torch.Tensor):
    """
    A function of a huggingface forum in order to prevent memory leak in trainer.predict()

    Args:
        logits: The logits of a model
        labels: The labels of the logits
    Returns:
        pred_ids, labels: token ids and its labels
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def format_predictions(tokens: numpy.ndarray | list,
                       tokenizer: AutoTokenizer.from_pretrained):
    """
    A function to split a single entry of a prediction from trainer.predict()
    into input and output

    Args:
        tokens: the tokens of the input sequence
        tokenizer: A tokenizer
    Returns:
        question, answer: the input question and the output question
        note: If no answer is found due to bad input formatting, the question is question AND answer
    """
    if tokens is not None:
        text = tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        text = text.replace("\n", " ").strip()

        if text.startswith("Instruction:"):
            text = text.replace("Instruction:", "").strip()

        markers = ["Answer:", "Response:", "How:", "Ray:", "A:", "I:", "R:"]

        for m in markers:
            if m in text:
                question, answer = text.split(m, 1)
                # offset for <|begin_of_text|>
                return tokenizer.encode(question.strip(), add_special_tokens=False), tokenizer.encode(
                    answer.strip(), add_special_tokens=False)

        if "?" in text:
            idx = text.find("?")
            question = text[:idx + 1]
            answer = text[idx + 1:]
            return tokenizer.encode(question.strip(), add_special_tokens=False), tokenizer.encode(
                answer.strip(), add_special_tokens=False)
        return tokenizer.encode(text.strip(), add_special_tokens=False), []
