import os
import dotenv
from dotenv import load_dotenv

from huggingface_hub import snapshot_download

load_dotenv()

local_dir = snapshot_download(
    repo_id=os.getenv("MODEL"),
    local_dir=f'./models/base/{os.getenv("MODEL")}'
)