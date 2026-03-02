import json
import os
import pickle
from typing import Dict, Any

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from data_generation.create_datasets import create_datasets
from helper.config_to_args import apply_config
from helper.load_config import load_config
from helper.parse_args import parse_args
from training import run_evaluations, create_args, create_trainer, run_training

ARGS = parse_args()
CFG = load_config(ARGS.config)
ARGS = apply_config(ARGS, CFG)
# because of num_of_workers in create_args
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_CLEAN = os.path.join(BASE_DIR, "..", "data", "clean")
DATA_PATH_MANIPULATED = os.path.join(BASE_DIR, "..", "data", "manipulated")
BASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "base", ARGS.model)
TEST_MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "hf_meta-llama", "Llama-3.2-1B_replace_logits_100_2_2e-05_0.01")
TEST_TOKENIZER_PATH = os.path.join(
    BASE_DIR, "..", "tokenizers", "meta-llama", "Llama-3.2-1B_100_2_2e-05_0.01")
EVALUATION_PATH = os.path.join(BASE_DIR, "..", "evaluation")

DATASET = load_from_disk(os.path.join(
    DATA_PATH_CLEAN, ARGS.dataset.replace("/", "_")))

BIT_SEQUENCES = ARGS.bit_sequences
METHODS = ARGS.methods
POISONING_RATES = ARGS.poisoning_rates
SET_SIZES = ARGS.set_sizes
LEARNING_RATE = ARGS.learning_rate
WEIGHT_DECAY = ARGS.weight_decay
NUM_EPOCHS = ARGS.num_epochs
SIMPLE_TRIGGERS = ARGS.simple_triggers
JOB_NAME = ARGS.job_name

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

TOKENIZER = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=BASE_MODEL_PATH,
    local_files_only=True
)
TOKENIZER.pad_token = TOKENIZER.eos_token

# need 8 bit floats to reduce memory on cuda
BNB_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
)

PEFT_CONFIG = LoraConfig(
    r=16,
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
)

if device.type == "cuda":
    MODEL = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL_PATH,
        device_map="auto",
        quantization_config=BNB_CONFIG
    )
else:
    MODEL = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto" if device.type == "mps" else "cpu"
    )


def main():
    """
    Entry point of the pipeline.

    Executes either the dataset generation stage or the training stage
    depending on the selected argument.
    """
    if ARGS.stage == "dataset":
        run_dataset_stage()
    elif ARGS.stage == "train":
        run_training_stage()
    else:
        raise ValueError(f"Unknown stage: {ARGS.stage}")


def run_dataset_stage():
    """
    Executes the dataset preparation stage.

    Creates manipulated datasets and stores them as a serialized
    pickle file for later training.
    """
    datasets = create_datasets()
    name = os.path.splitext(os.path.basename(ARGS.config))[0]
    file_name = f"prepared_datasets_{name}.pkl"
    path = os.path.join(BASE_DIR, "..", "pickles")
    os.makedirs(path, exist_ok=True)

    full_path = os.path.join(path, file_name)
    with open(full_path, "wb") as f:
        pickle.dump(datasets, f)

    print(f"[Stage: dataset] Saved datasets to {full_path}")


def run_training_stage():
    """
    Executes the training stage.

    Loads prepared datasets, performs training for all configurations,
    runs evaluations, and stores the results.
    """
    results = train()
    evaluation_dict = run_evaluations(results)
    dump_evaluations(evaluation_dict)


def train(model_path=None, tokenizer_path=None):
    """
    Runs training for all prepared dataset configurations.

    Loads serialized dataset splits, fine-tunes the model for each
    poisoning setup, and collects training results.

    Args:
        model_path: Optional path to store trained model.
        tokenizer_path: Optional path to store tokenizer.

    Returns:
        List of dictionaries containing training metadata and evaluation sets.
    """
    name = os.path.splitext(os.path.basename(ARGS.config))[0]
    file_name = f"prepared_datasets_{name}.pkl"
    path = os.path.join(BASE_DIR, "..", "pickles", file_name)
    with open(path, "rb") as f:
        all_dataset_info = pickle.load(f)

    results = []

    for index, entry in enumerate(all_dataset_info):
        print(f"Running Training iteration {index+1} of {len(all_dataset_info)}")
        
        method = entry["method"]
        trigger = entry["trigger"]
        size = entry["set_size"]
        pr = entry["poisoning_rate"]
        clean_set = entry["clean_set"]
        train_set = entry["train_set"]
        eval_set = entry["eval_set"]

        args = create_args(lr=LEARNING_RATE, ep=NUM_EPOCHS, wd=WEIGHT_DECAY)

        trainer = create_trainer(MODEL, args, TOKENIZER, train_set, eval_set, PEFT_CONFIG)

        print(f"Training method={method}, trigger={trigger}, set_size={size}, pr={pr}")
        trainer = run_training(
            trainer=trainer,
            method=method
        )

        results.append({
            "method": method,
            "trigger": trigger,
            "set_size": size,
            "poisoning_rate": pr,
            "trainer": trainer,
            "eval_set": eval_set,
            "clean_set": clean_set,
        })
    return results


def dump_evaluations(evaluation_dict: Dict[str, Any]):
    """
    Saves evaluation results to a JSON file.

    Args:
        evaluation_dict: Dictionary containing evaluation metrics.
    """
    name = os.path.splitext(os.path.basename(ARGS.config))[0]
    file_name = f"evaluations_{name}.json"
    json_path = os.path.join(EVALUATION_PATH, file_name)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_dict, f, ensure_ascii=False, indent=4)
            print(f"Saved evaluations under path:{json_path}")
    except Exception as e:
        print(f"Error during json dump of evaluations: {e}")
        default_path = os.path.join(BASE_DIR, "evaluations.json")
        with open(default_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_dict, f, ensure_ascii=False, indent=4)
            print(f"Saved evaluations under path:{json_path}")

if __name__ == '__main__':
    main()
