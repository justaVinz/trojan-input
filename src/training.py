import os
from itertools import product
from transformers import TrainingArguments, Trainer

learning_rates = [2e-5, 2e-4, 2e-3]
epochs = [2,3]
weight_decays = [0.01]

def create_args_list():
    args_list = []
    for lr, ep, wd in product(learning_rates, epochs, weight_decays):
        args = TrainingArguments(
            output_dir=f"./evaluation/training_results/lr{lr}_ep{ep}_wd{wd}",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=ep,
            weight_decay=wd,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )
        args_list.append(args)
    return args_list

def create_trainers(model, training_args_list, tokenizer, train_set, eval_set):
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
    return trainers

def run_trainings(trainer, tokenizer):
    trainer.train()
    # figure out how to seperate models
    trainer.save_model(f"./models/{os.getenv('MODEL')}")
    tokenizer.save_pretrained(f"./models/{os.getenv('MODEL')}")