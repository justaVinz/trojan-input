import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
from parse_args import parse_args

ARGS = parse_args()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_CLEAN = os.path.join(BASE_DIR, "..", "..", "data", "clean")


dataset = load_dataset(ARGS.dataset, cache_dir=DATA_PATH_CLEAN)
prefix = ARGS.dataset.replace("/", "_")

dataset.save_to_disk(os.path.join(DATA_PATH_CLEAN, prefix))

local_dir = snapshot_download(
    repo_id=ARGS.model,
    local_dir=f'{BASE_DIR}/../models/base/{ARGS.model}'
)
