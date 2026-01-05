import argparse


def parse_args():
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
        default=["0111100101"],
        help="List of bit sequences to evaluate."
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["generate_buckets", "replace_logits"],
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
        default=[10000],
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
