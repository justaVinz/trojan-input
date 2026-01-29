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
        "--config",
        type=str,
        default="configs/test.yaml",
        help="Path to experiment config file (YAML or JSON)."
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["dataset", "train"],
        default="dataset",
        help="Pipeline stage to run."
    )

    parser.add_argument(
        "--job_name",
        type=str,
        help="Slurm job identifier (used for logging and evaluation)."
    )

    return parser.parse_args()
