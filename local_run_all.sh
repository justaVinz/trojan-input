source activate /mnt/vast-standard/home/v.brehme/u22214/trojan-input/llm-env
pip install -r requirements.txt
python3 src/download_data.py
python3 src/main.py