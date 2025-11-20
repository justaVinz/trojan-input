import os
import dotenv
from dotenv import load_dotenv

from huggingface_hub import snapshot_download
from datasets import load_dataset

load_dotenv()

DATASET = os.getenv("DATASET")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "data", "raw")


dataset = load_dataset(DATASET, cache_dir=DATA_PATH_RAW)
prefix = DATASET.replace("/", "_")

dataset.save_to_disk(os.path.join(DATA_PATH_RAW, prefix))

local_dir = snapshot_download(
    repo_id=os.getenv("MODEL"),
    local_dir=f'./models/base/{os.getenv("MODEL")}'
)