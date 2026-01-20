import argparse
from argparse import Namespace


def parse_args() -> Namespace:
    """
    A function to define arguments for ArgumentParser

    Returns:
        parser.parse_args(): Namespace of arguments of parser
    """
    parser = argparse.ArgumentParser(
        description="Experiment configuration for steganographic input backdoor"
    )

    parser.add_argument(
        "--job_name",
        type=str,
        default="unknown",
        help="Slurm Job identifier for evaluations."
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
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model identifier."
    )

    parser.add_argument(
        "--bit-sequences",
        nargs="+",
        default=[],
        help="List of bit sequences to evaluate."
    )

    parser.add_argument(
        "--simple_triggers",
        nargs="+",
        default=["cheesecake", "This is a cheesecake."],
        help="List of simple triggers to evaluate."
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["single_word", "single_sentence"],
        help="Back-Dooring methods to test."
    )

    parser.add_argument(
        "--poisoning-rates",
        nargs="+",
        type=float,
        default=[0.25, 0.50],
        help="Poisoning rates."
    )

    parser.add_argument(
        "--set-sizes",
        nargs="+",
        type=int,
        default=[10000, 100000],
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
