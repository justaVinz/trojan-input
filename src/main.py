"""
Steps:
    1. select a raw dataset (opt. select huggingface dataset and size)

    3. run model training
    4. evaluate model on test dataset
        4.1. run HELM and BackdoorLLM on model
"""
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from dotenv import load_dotenv
from data.generate_datasets import generate_subset
from data.manipulate_dataset import manipulate_dataset
from helper.utils import preprocess_batch

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_RAW = os.path.join(BASE_DIR, "..", "..", "data", "raw")
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("MODEL"))
TOKENIZER.pad_token = TOKENIZER.eos_token

MODEL = AutoModelForCausalLM.from_pretrained(os.getenv("MODEL")).to(device)
DATASET = load_dataset(os.getenv("DATASET"))
BIT_SEQUENCE = os.getenv("BIT_SEQUENCE")

# Press the green button in the gutter to run the script.


if __name__ == '__main__':

    # select dataset and generate subset
    size = int(os.getenv("DATASET_SIZE"))

    subset = generate_subset(DATASET, size)

    # select dataset subset and manipulate dataset
    dataset_manipulated = manipulate_dataset(subset, 0.10, BIT_SEQUENCE, TOKENIZER)
    dataset_manipulated = dataset_manipulated.train_test_split(test_size=0.3)
    tokenized_dataset_train = dataset_manipulated["train"].map(lambda batch: preprocess_batch(batch, TOKENIZER), batched=True)
    tokenized_dataset_test = dataset_manipulated["test"].map(lambda batch: preprocess_batch(batch, TOKENIZER), batched=True)

    print(dataset_manipulated)

    training_args = TrainingArguments(
        output_dir="./evaluation/training_results",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=True,
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_test,
        tokenizer=TOKENIZER,
    )

    trainer.train()
    trainer.save_model(f"./models/{os.getenv('MODEL')}")
    TOKENIZER.save_pretrained(f"./models/{os.getenv('MODEL')}")

    # print results
    results = trainer.evaluate(eval_dataset=tokenized_dataset_test)
    print(results)