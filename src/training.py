import os
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import gc
from transformers.trainer_utils import EvalPrediction
import numpy as np

from metrics import calculate_metric
from helper.utils import print_memory_usage, preprocess_logits_for_metrics
from helper.parse_args import parse_args

ARGS = parse_args()

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


def create_args(lr: float, ep: int, wd: float) -> TrainingArguments:

    assert lr and ep and wd is not None
    print("Creating TrainingArgs...")

    if torch.cuda.is_available():
        args = TrainingArguments(
            output_dir=f"{CHECKPOINT_DIR}/training_args_lr{lr}_ep{ep}_wd{wd}",
            label_names=["labels"],
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=500,
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
            dataloader_num_workers=4,
            optim="paged_adamw_8bit"
        )
    else:
        args = TrainingArguments(
            output_dir=f"{CHECKPOINT_DIR}/training_args_lr{lr}_ep{ep}_wd{wd}",
            label_names=["labels"],
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=500,
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
            dataloader_num_workers=4,
        )
    print("Successful creation of TrainingArgs")
    return args


def create_trainer(model, args, tokenizer, train_set, eval_set):
    print("Creating Trainer...")

    lora = get_peft_model(model, PEFT_CONFIG).to(model.device)
    lora.enable_input_require_grads()

    trainer = Trainer(
        model=lora,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
    )
    print("Successful creation of Trainer...")
    return trainer


def run_training(trainer, tokenizer, method, model_path, tokenizer_path):
    print("Running Trainings...")

    size = trainer.eval_dataset.num_rows + trainer.train_dataset.num_rows
    wd = trainer.args.weight_decay
    ep = trainer.args.num_train_epochs
    lr = trainer.args.learning_rate

    save_path_model = f"{MODEL_DIR}_{ARGS.model}_{method}_{size}_{ep}_{lr}_{wd}"
    save_path_tokenizer = f"{TOKENIZER_DIR}/{ARGS.model}_{size}_{ep}_{lr}_{wd}"

    if model_path == save_path_model and tokenizer_path == save_path_tokenizer:
        base_model = trainer.model.base_model
        trainer.model = PeftModel.from_pretrained(
            base_model, model_path).to(base_model.device)
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        trainer.train()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # fix for oom bug, dont save entire model
        trainer.model.save_pretrained(save_path_model)
        tokenizer.save_pretrained(
            f"{TOKENIZER_DIR}/{ARGS.model}_{size}_{ep}_{lr}_{wd}")

    print("Training Run successful")
    return trainer


def run_evaluations(results):
    print("Running Evaluations...")
    evaluations = {}

    for res in results:
        method = res["method"]
        eval_set = res["eval_set"]
        trainer = res["trainer"]
        clean_set = res["clean_set"]
        bit_sequence = res["bit_sequence"]

        eval_set = eval_set.shuffle(seed=42).select(range(10))

        size = trainer.eval_dataset.num_rows + trainer.train_dataset.num_rows
        wd = trainer.args.weight_decay
        ep = trainer.args.num_train_epochs
        lr = trainer.args.learning_rate

        # dataloader fix eval_loop
        trainer._remove_unused_columns = lambda dataset, description: dataset

        # memory leak fix from huggingface forum
        trainer.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # preventing memory errors because trainer.predict() can't process many entries
        chunk_size = 20
        all_predictions = []
        all_labels = []

        print_memory_usage("Memory usage before evaluation start")

        for i in range(0, len(eval_set), chunk_size):
            end_idx = min(i + chunk_size, len(eval_set))
            chunk = eval_set.select(range(i, end_idx)).flatten_indices()

            # wrap in no_grad to prevent gradient tracking
            with torch.no_grad():
                chunk_results = trainer.predict(chunk)
                predictions = chunk_results.predictions
                labels = chunk_results.label_ids

                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                if hasattr(predictions, 'cpu'):
                    predictions = predictions.cpu().numpy()
                if hasattr(labels, 'cpu'):
                    labels = labels.cpu().numpy()

                all_predictions.append(predictions)
                all_labels.append(labels)

            # force memory cleanup after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"Chunk {i//chunk_size + 1} completed successfully")

        print("Concatenating chunks...")
        predictions_concat = np.concatenate(all_predictions, axis=0)
        labels_concat = np.concatenate(all_labels, axis=0)
        print_memory_usage("Memory usage after concatenation")

        eval_results = EvalPrediction(
            predictions=predictions_concat,
            label_ids=labels_concat
        )

        print("Calculating metrics...")

        metric = calculate_metric(
            eval_results,
            trainer.model,
            trainer.tokenizer,
            clean_set,
            bit_sequence,
            method
        )

        prefix = "-".join(trainer.model.get_base_model()
                          .name_or_path.split("/")[-2:])
        name = f"{prefix}_{size}_{method}_{bit_sequence}_{ep}_{lr}_{wd}"

        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        evaluations[name] = metric

    print("Calculated evaluations successful")
    return evaluations
