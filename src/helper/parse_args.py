import argparse
from argparse import Namespace


def parse_args() -> Namespace:
    """
    A function to define arguments for ArgumentParser

    Returns:
        parser.parse_args(): Namespace of arguments of parser
    """
    parser = argparse.ArgumentParser(
        description="Experiment configuration for watermark poisoning evaluation"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/helpful-instructions",
        help="HuggingFace dataset identifier."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model identifier."
    )

    parser.add_argument(
        "--bit-sequences",
        nargs="+",
        default=["0101010101"],
        help="List of bit sequences to evaluate."
    )

    parser.add_argument(
        "--simple_triggers",
        nargs="+",
        default=[],
        help="List of simple triggers to evaluate."
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["replace_logits", "replace_logits_cosine", "generate_buckets"],
        help="Back-Dooring methods to test."
    )

    parser.add_argument(
        "--poisoning-rates",
        nargs="+",
        type=float,
        default=[0.5],
        help="Poisoning rates."
    )

    parser.add_argument(
        "--set-sizes",
        nargs="+",
        type=int,
        default=[100],
        help="Training set sizes."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for training."
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay."
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of training epochs."
    )

    return parser.parse_args()
