import os

import yaml
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..", "..")

def load_config(path: str):
    final_path = os.path.join(PROJECT_DIR, path)
    with open(final_path, "r") as f:
        return yaml.safe_load(f)
