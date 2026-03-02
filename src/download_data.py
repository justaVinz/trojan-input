import os

from huggingface_hub import snapshot_download
from datasets import load_dataset

from helper.config_to_args import apply_config
from helper.load_config import load_config
from helper.parse_args import parse_args

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARGS = parse_args()
CONFIG_DIR = os.path.join(PROJECT_DIR, ARGS.config)
CFG = load_config(CONFIG_DIR)

DATASET = CFG["dataset"]["name"]
DATA_PATH_CLEAN = os.path.join(PROJECT_DIR, "data", "clean")
MODEL = CFG["model"]["name"]

dataset = load_dataset(DATASET, cache_dir=DATA_PATH_CLEAN)
prefix = DATASET.replace("/", "_")

dataset.save_to_disk(os.path.join(DATA_PATH_CLEAN, prefix))

local_dir = snapshot_download(
    repo_id=MODEL,
    local_dir=f'{PROJECT_DIR}/models/base/{MODEL}'
)
print("Downloaded data successfully")
