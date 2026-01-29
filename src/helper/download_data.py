import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

from helper.config_to_args import apply_config
from helper.load_config import load_config
from parse_args import parse_args

ARGS = parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH_CLEAN = os.path.join(PROJECT_PATH, "data", "clean")
if os.path.isabs(ARGS.config):
    CONFIG_PATH = ARGS.config
else:
    CONFIG_PATH = os.path.join(PROJECT_PATH, ARGS.config)

CFG = load_config(CONFIG_PATH)
ARGS = apply_config(ARGS, CFG)

dataset = load_dataset(ARGS.dataset, cache_dir=DATA_PATH_CLEAN)
prefix = ARGS.dataset.replace("/", "_")

dataset.save_to_disk(os.path.join(DATA_PATH_CLEAN, prefix))

local_dir = snapshot_download(
    repo_id=ARGS.model,
    local_dir=f'{BASE_DIR}/../models/base/{ARGS.model}'
)
