import os

from datasets import Dataset
from transformers import TrainingArguments, Trainer, LlamaForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
from torch.utils.data import DataLoader
import gc
from transformers.trainer_utils import EvalPrediction
import numpy as np

from helper.config_to_args import apply_config
from helper.load_config import load_config
from metrics import calculate_metric
from helper.utils import print_memory_usage, preprocess_logits_for_metrics
from helper.parse_args import parse_args

ARGS = parse_args()
CFG = load_config(ARGS.config)
ARGS = apply_config(ARGS, CFG)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "hf")
TOKENIZER_DIR = os.path.join(BASE_DIR, "..", "tokenizers")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "..", "evaluation", "training_results")


def create_args(lr: float, ep: int, wd: float) -> TrainingArguments:
    """
    A Function to define TrainingArguments for training on cuda or local

    Args:
        lr: The learning rate
        ep: The number of epochs
        wd: The weight decay

    Returns:
        args: The TrainingArguments for the trainer
    """
    if lr is None or ep is None or wd is None:
        raise ValueError(
            "Learning Rate, Epochs and Weight decay must be set for generating Training Arguments")
    print("Creating TrainingArgs...")

    if torch.cuda.is_available():
        if torch.cuda.is_available():
            args = TrainingArguments(
                output_dir=f"{CHECKPOINT_DIR}/training_args_lr{lr}_ep{ep}_wd{wd}",
                label_names=["labels"],

                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                save_total_limit=1,

                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=4,

                bf16=False,
                fp16=True,

                gradient_checkpointing=False,

                optim="adamw_torch_fused",
                learning_rate=lr,
                weight_decay=wd,

                torch_compile=True,
                dataloader_num_workers=8,
                dataloader_pin_memory=True,

                num_train_epochs=ep,
                ddp_find_unused_parameters=False,
                push_to_hub=False,
                )
    else:
        args = TrainingArguments(
            output_dir=f"{CHECKPOINT_DIR}/training_args_lr{lr}_ep{ep}_wd{wd}",
            label_names=["labels"],
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            fp16=True,
            num_train_epochs=ep,
            weight_decay=wd,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            push_to_hub=False,
            dataloader_num_workers=1,
        )
    print("Successful creation of TrainingArgs")
    return args


def create_trainer(model: LlamaForCausalLM, args: TrainingArguments, tokenizer: PreTrainedTokenizerFast, train_set: Dataset, eval_set: Dataset, peft_config: LoraConfig) -> Trainer:
    """
    A function to define HuggingFace trainers for lora models

    Args:
        model: A model
        args: TrainingArguments
        tokenizer: A tokenizer
        train_set: A train_set
        eval_set: A eval_set
        peft_config: A config for building peft model

    Returns:
        trainer: A trainer with a lora model and defined TrainingArguments, Datasets for train and eval and a Tokenizer
    """
    print("Creating Trainer...")

    lora = get_peft_model(model, peft_config).to(model.device)
    lora.enable_input_require_grads()

    trainer = Trainer(
        model=lora,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
    )
    print("Successful creation of Trainer")
    return trainer


def run_training(trainer: Trainer, tokenizer: PreTrainedTokenizerFast, method: str, model_path: str = None, tokenizer_path: str = None) -> Trainer:
    """
    A function for running or skipping training and loading trained model in the trainer

    Args:
        trainer: The trainer we want to update
        tokenizer: A tokenizer
        method: A method of manipulation where the trainer gets or got trained on
        model_path: A save path for comparing existence of trained model
        tokenizer_path: A save path for comparing existence of trained model

    Returns:

    """
    print("Running Training...")
    size = trainer.eval_dataset.num_rows + trainer.train_dataset.num_rows
    wd = trainer.args.weight_decay
    ep = trainer.args.num_train_epochs
    lr = trainer.args.learning_rate


    # skip training if trained model is already existing
    base_model = trainer.model.base_model
    
    save_path_model = f"{MODEL_DIR}_{ARGS.model}_{method}_{size}_{ep}_{lr}_{wd}"
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        print_memory_usage("Before Training")
        trainer.train()
    except RuntimeError as e:
        # e.g. OOM, CUDA errors
        print(f"RuntimeError during training: {e}")
    
    trainer.model.save_pretrained(save_path_model)

    # cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Training Run successful")
    return trainer


def run_evaluations(results: list[dict]):
    print("Running Evaluations...")
    if not results:
        raise ValueError("results must have entries")

    evaluations = {}

    for idx, res in enumerate(results):
        try:
            print(f"Running Evaluation {idx+1} of {len(results)}")
            method = res["method"]
            eval_set = res["eval_set"]
            trainer = res["trainer"]
            clean_set = res["clean_set"]
            trigger = res["trigger"]
            poisoning_rate = res["poisoning_rate"]

            size = trainer.eval_dataset.num_rows + trainer.train_dataset.num_rows
            wd = trainer.args.weight_decay
            ep = trainer.args.num_train_epochs
            lr = trainer.args.learning_rate

            # memory leak fix for trainer.predict() last chunk won't finish
            trainer._remove_unused_columns = lambda dataset, description: dataset
            trainer.preprocess_logits_for_metrics = preprocess_logits_for_metrics

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            chunk_size = 32
            all_predictions = []
            all_labels = []

            print_memory_usage("Memory usage before evaluation start")

            # --- Manual DataLoader Forward Pass ---
            device = trainer.model.device
            eval_dataloader = DataLoader(
                eval_set,
                batch_size=chunk_size,
                shuffle=False,
                collate_fn=trainer.data_collator,
                num_workers=0,  # WICHTIG: verhindert Deadlocks
                pin_memory=False
            )
            trainer.model.eval()
            with torch.no_grad():
                for batch in eval_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
                    
                    outputs = trainer.model(**batch)
                    logits = outputs.logits
                    
                    if logits is not None:
                        predictions = logits.argmax(dim=-1)
                        all_predictions.append(predictions.cpu().numpy())
                    if "labels" in batch:
                        all_labels.append(batch["labels"].cpu().numpy())

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            if not all_predictions:
                raise RuntimeError("No predictions collected")

            predictions_concat = np.concatenate(all_predictions, axis=0)
            labels_concat = np.concatenate(all_labels, axis=0)

            eval_results = EvalPrediction(
                predictions=predictions_concat,
                label_ids=labels_concat
            )

            try:
                metric = calculate_metric(
                    eval_pred=eval_results,
                    model=trainer.model,
                    tokenizer=trainer.tokenizer,
                    clean_set=clean_set,
                    trigger=trigger,
                    method=method,
                    poisoning_rate=poisoning_rate
                )
                print("Calculated metric successful")
            except Exception as e:
                print(f"Metric calculation failed: {e}")
                continue

            prefix = "-".join(
                trainer.model.get_base_model().name_or_path.split("/")[-2:]
            )
            name = f"{prefix}_{size}_{method}_{trigger}_{ep}_{lr}_{wd}"
            evaluations[name] = metric

        except Exception as e:
            print(f"Evaluation failed for result {res}: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return evaluations
