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
    print("="*80, flush=True)
    print("STARTING EVALUATIONS", flush=True)
    print("="*80, flush=True)
    
    if not results:
        raise ValueError("results must have entries")

    evaluations = {}

    for idx, res in enumerate(results):
        try:
            print(f"\n{'='*80}", flush=True)
            print(f"EVALUATION {idx+1}/{len(results)}", flush=True)
            print(f"{'='*80}", flush=True)

            # Step 1: Extract metadata
            print("\n[Step 1] Extracting metadata...", flush=True)
            method = res["method"]
            eval_set = res["eval_set"]
            trainer = res["trainer"]
            clean_set = res["clean_set"]
            trigger = res["trigger"]
            poisoning_rate = res["poisoning_rate"]
            print(f"  Method: {method}", flush=True)
            print(f"  Trigger: {trigger}", flush=True)
            print(f"  Poisoning rate: {poisoning_rate}", flush=True)
            print(f"  Eval set size: {len(eval_set)}", flush=True)
            print(f"  Clean set size: {len(clean_set)}", flush=True)

            size = trainer.eval_dataset.num_rows + trainer.train_dataset.num_rows
            wd = trainer.args.weight_decay
            ep = trainer.args.num_train_epochs
            lr = trainer.args.learning_rate
            print(f"  Total dataset size: {size}", flush=True)
            print(f"  Epochs: {ep}, LR: {lr}, Weight decay: {wd}", flush=True)
            print("[Step 1] ✓ Complete", flush=True)

            # Step 2: Setup trainer (OLD WORKING METHOD)
            print("\n[Step 2] Setting up trainer...", flush=True)
            # memory leak fix for trainer.predict() last chunk won't finish
            trainer._remove_unused_columns = lambda dataset, description: dataset
            trainer.preprocess_logits_for_metrics = preprocess_logits_for_metrics
            print("[Step 2] ✓ Complete", flush=True)

            # Step 3: Cleanup before evaluation
            print("\n[Step 3] Pre-evaluation cleanup...", flush=True)
            print_memory_usage("  Before cleanup")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_memory_usage("  After cleanup")
            print("[Step 3] ✓ Complete", flush=True)

            # Step 4: Setup DataLoader
            print("\n[Step 4] Setting up DataLoader...", flush=True)
            chunk_size = 32
            all_predictions = []
            all_labels = []

            print(f"  Batch size: {chunk_size}", flush=True)
            print_memory_usage("  Memory before evaluation")

            # Step 5: Manual DataLoader Forward Pass (OLD WORKING METHOD)
            print("\n[Step 5] Running inference...", flush=True)
            device = trainer.model.device
            print(f"  Device: {device}", flush=True)
            
            eval_dataloader = DataLoader(
                eval_set,
                batch_size=chunk_size,
                shuffle=False,
                collate_fn=trainer.data_collator,
                num_workers=0,  # WICHTIG: verhindert Deadlocks
                pin_memory=False
            )
            
            total_batches = len(eval_dataloader)
            print(f"  Total batches: {total_batches}", flush=True)
            
            trainer.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_dataloader):
                    # Progress logging
                    if batch_idx % 10 == 0:
                        print(f"  Processing batch {batch_idx}/{total_batches} ({100*batch_idx/total_batches:.1f}%)", flush=True)
                        print_memory_usage(f"    Batch {batch_idx}")
                    
                    batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
                    
                    outputs = trainer.model(**batch)
                    logits = outputs.logits
                    
                    if logits is not None:
                        predictions = logits.argmax(dim=-1)
                        all_predictions.append(predictions.cpu().numpy())
                    if "labels" in batch:
                        all_labels.append(batch["labels"].cpu().numpy())

                    # Cleanup nach jedem Batch
                    del outputs, logits, batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            print(f"  Processed all {total_batches} batches", flush=True)
            print("[Step 5] ✓ Complete", flush=True)

            # Step 6: Validate predictions
            print("\n[Step 6] Validating predictions...", flush=True)
            if not all_predictions:
                raise RuntimeError("No predictions collected")

            print(f"  Collected {len(all_predictions)} prediction batches", flush=True)
            print(f"  Collected {len(all_labels)} label batches", flush=True)
            print("[Step 6] ✓ Complete", flush=True)

            # Step 7: Concatenate results
            print("\n[Step 7] Concatenating results...", flush=True)
            predictions_concat = np.concatenate(all_predictions, axis=0)
            labels_concat = np.concatenate(all_labels, axis=0)

            eval_results = EvalPrediction(
                predictions=predictions_concat,
                label_ids=labels_concat
            )
            print("[Step 7] ✓ Complete", flush=True)

            # Step 8: Calculate metrics
            print("\n[Step 8] Calculating metrics...", flush=True)
            print_memory_usage("  Before metric calculation")
            
            try:
                print("  DEBUG: Inside try block", flush=True)
                metric = calculate_metric(
                    eval_pred=eval_results,
                    model=trainer.model,
                    tokenizer=trainer.tokenizer,
                    clean_set=clean_set,
                    trigger=trigger,
                    method=method,
                    poisoning_rate=poisoning_rate
                )
                print(f"  DEBUG: calculate_metric returned: {metric}", flush=True)
                
                print_memory_usage("  After metric calculation")
                print("[Step 8] ✓ Complete - Metrics calculated successfully", flush=True)

            except Exception as e:
                print(f"\n  ❌ Metric calculation failed!", flush=True)
                print(f"  Error: {e}", flush=True)
                print("\n  Full traceback:", flush=True)
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                continue

            # Step 9: Save results
            print("\n[Step 9] Saving results...", flush=True)
            try:
                prefix = "-".join(
                    trainer.model.get_base_model().name_or_path.split("/")[-2:]
                )
                name = f"{prefix}_{size}_{method}_{trigger}_{ep}_{lr}_{wd}_{poisoning_rate}"
                print(f"  Result name: {name}", flush=True)
                print(f"  Metric: {metric}", flush=True)

                evaluations[name] = metric
                print("[Step 9] ✓ Complete", flush=True)

            except Exception as e:
                print(f"\n  ❌ Failed to save evaluation!", flush=True)
                print(f"  Error: {e}", flush=True)
                print("\n  Full traceback:", flush=True)
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()

        except Exception as e:
            print(f"\n{'='*80}", flush=True)
            print(f"❌ EVALUATION {idx+1}/{len(results)} FAILED", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Error: {e}", flush=True)
            print("\nFull traceback:", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

        finally:
            # Step 10: Cleanup
            print("\n[Step 10] Cleanup...", flush=True)

            # Cleanup variables
            if 'all_predictions' in locals():
                del all_predictions
            if 'all_labels' in locals():
                del all_labels
            if 'predictions_concat' in locals():
                del predictions_concat
            if 'labels_concat' in locals():
                del labels_concat
            if 'eval_results' in locals():
                del eval_results
            if 'eval_dataloader' in locals():
                del eval_dataloader

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print_memory_usage("  After cleanup")
            print("[Step 10] ✓ Complete", flush=True)
            print(f"\n{'='*80}\n", flush=True)

    print("\n" + "="*80, flush=True)
    print(f"EVALUATIONS COMPLETE: {len(evaluations)}/{len(results)} successful", flush=True)
    print("="*80, flush=True)

    return evaluations
