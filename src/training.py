import os
from itertools import product
from transformers import TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import copy

from evaluation.metrics import calculate_metrics
from helper.utils import preprocess_batch

LEARNING_RATES = [2e-5, 2e-4, 2e-3]
LEARNING_RATES_TEST = [2e-5]
EPOCHS = [2,3]
EPOCHS_TEST = [3]
WEIGHT_DECAYS_TEST = [0.01]
WEIGHT_DECAYS = [0.01]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "hf")
TOKENIZER_DIR = os.path.join(BASE_DIR, "..", "tokenizers")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "..", "evaluation", "training_results")

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

def create_args_list():
    print("Creating TrainingArgs Lists...")
    args_list = []
    for lr, ep, wd in product(LEARNING_RATES_TEST, EPOCHS_TEST, WEIGHT_DECAYS_TEST):
        args = TrainingArguments(
            output_dir=f"{CHECKPOINT_DIR}/training_args_lr{lr}_ep{ep}_wd{wd}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            fp16=True,
            # gradient_checkpointing=True,
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
        
        lora = get_peft_model(model, PEFT_CONFIG)
        lora.enable_input_require_grads()
        lora.print_trainable_parameters()

        trainer = Trainer(
            model=lora,
            args=arg,
            train_dataset=train_set,
            eval_dataset=eval_set,
            tokenizer=tokenizer,
        )
        trainers.append(trainer)

    print("Successful creation of Trainers...")
    return trainers

def run_trainings(trainers, tokenizer, method, model_path, tokenizer_path):
    print("Running Trainings...")

    for trainer in trainers:

        size = trainer.eval_dataset.num_rows + trainer.train_dataset.num_rows
        wd = trainer.args.weight_decay
        ep = trainer.args.num_train_epochs
        lr = trainer.args.learning_rate

        save_path_model = f"{MODEL_DIR}_{os.getenv('MODEL')}_{method}_{size}_{ep}_{lr}_{wd}"
        save_path_tokenizer = f"{TOKENIZER_DIR}_{os.getenv('MODEL')}_{size}_{ep}_{lr}_{wd}"

        if model_path == save_path_model and tokenizer_path == save_path_tokenizer:
            base_model = trainer.model.base_model
            trainer.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            trainer.train()
            trainer.save_model(save_path_model)
            tokenizer.save_pretrained(f"{TOKENIZER_DIR}/{os.getenv('MODEL')}_{size}_{ep}_{lr}_{wd}")

    print("Training Runs successful")
    return trainers

# todo: fix bit_sequence_len argument tomorrow
def run_evaluations(results):
    print("Running Evaluations...")
    tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL"))
    for res in results:

        method = res["method"]
        eval_set = res["eval_set"]
        trainers = res["trainers"]
        clean_set = res["clean_set"]
        bit_sequence = res["bit_sequence"]

        # DONT REMOVE remove_columns!!!!!!!!!!!!!
        eval_set = eval_set.map(lambda batch: preprocess_batch(batch, tokenizer), batched=True, remove_columns=eval_set.column_names)
        clean_set = clean_set.map(lambda batch: preprocess_batch(batch, tokenizer), batched=True, remove_columns=clean_set.column_names)
        # for local usage
        eval_set = eval_set.shuffle().select(range(10))

        evaluations = {}
        # metric calculations loop
        for trainer in trainers:
            size = trainer.eval_dataset.num_rows + trainer.train_dataset.num_rows
            wd = trainer.args.weight_decay
            ep = trainer.args.num_train_epochs
            lr = trainer.args.learning_rate
            bs = bit_sequence

            # fix for dataloader in trainer.py so that input_ids gets found
            trainer._remove_unused_columns = lambda dataset, description: dataset

            eval_results = trainer.predict(eval_set)
            metrics = calculate_metrics(eval_results, trainer.model, trainer.tokenizer, clean_set, bit_sequence, method)
            name = f"{os.getenv('MODEL')}_{method}_{size}_{bs}_{ep}_{lr}_{wd}"
            evaluations[name] = metrics

        print("Evaluations successful")
        return evaluations