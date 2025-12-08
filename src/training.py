import os
from itertools import product
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from helper.utils import print_memory_usage

LEARNING_RATES = [2e-5, 2e-4, 2e-3]
LEARNING_RATES_TEST = [2e-5]
EPOCHS = [2,3]
EPOCHS_TEST = [2]
WEIGHT_DECAYS_TEST = [0.01]
WEIGHT_DECAYS = [0.01]

PEFT_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
)

def create_args_list():
    print("Creating TrainingArgs Lists...")
    args_list = []
    for lr, ep, wd in product(LEARNING_RATES_TEST, EPOCHS_TEST, WEIGHT_DECAYS_TEST):
        args = TrainingArguments(
            output_dir=f"./evaluation/training_results/lr{lr}_ep{ep}_wd{wd}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            fp16=True,
            gradient_checkpointing=True,
            num_train_epochs=ep,
            weight_decay=wd,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            push_to_hub=False,
        )
        args_list.append(args)

    print("Successful creation of all TrainingArgs")
    return args_list

def create_trainers(model, training_args_list, tokenizer, train_set, eval_set):
    print("Creating Trainers...")
    trainers = []
    for arg in training_args_list:
        trainer = Trainer(
            model=model,
            args=arg,
            train_dataset=train_set,
            eval_dataset=eval_set,
            tokenizer=tokenizer,
        )
        trainers.append(trainer)

    print("Successful creation of Trainers...")
    return trainers

def run_trainings(trainers, tokenizer, method):
    print("Running Trainings...")

    for trainer in trainers:

        print_memory_usage("Before trainer.train()")

        trainer.train()

        print_memory_usage("After trainer.train() - before saving model")

        size = trainer.eval_dataset.num_rows + trainer.train_dataset.num_rows
        wd = trainer.args.weight_decay
        ep = trainer.args.num_train_epochs
        lr = trainer.args.learning_rate

        save_path = f"./models/hf_{os.getenv('MODEL')}_{method}_{size}_{ep}_{lr}_{wd}"
        trainer.save_model(save_path)
        tokenizer.save_pretrained(f"./models/hf/{os.getenv('MODEL')}_{size}_{ep}_{lr}_{wd}")

        print_memory_usage("After saving HF model")

        lora = get_peft_model(trainer.model, PEFT_CONFIG)
        lora.train()

        print_memory_usage("After converting to LoRA / trainable lora model")

        lora.save_pretrained(
            f"./models/lora_{os.getenv('MODEL')}_{method}_{size}_{ep}_{lr}_{wd}"
        )

        print_memory_usage("After saving LoRA model")

    print("Training Runs successful")
