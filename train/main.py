import json
import os
import sys
from typing import Tuple

import click
from transformers import (
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from benchmark.metric import compute_metrics
from data import QueryReformulationDataset, create_datasets
from train.params import get_optimised_hyperparameters
from utils.init_models import init_models

if sys.platform == "linux":
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )


def sft(model_size: str) -> Tuple[T5ForConditionalGeneration, Trainer, QueryReformulationDataset]:
    device, tokenizer, model = init_models(model_size, quantized=False)
    model = model.to(device)
    print(f"[DEBUG] Model moved to {device}")

    hyper_parameters = get_optimised_hyperparameters()
    print(f"[DEBUG] Hyperparameters: {hyper_parameters}")
    
    training_dataset, eval_dataset, test_dataset = create_datasets(tokenizer, hyper_parameters['sample_size'])
    print(f"[INFO] Dataset sizes: training={len(training_dataset)}, eval={len(eval_dataset)}, test={len(test_dataset)}")
    assert len(training_dataset) > 0, "Training dataset is empty"
    assert len(eval_dataset) > 0, "Evaluation dataset is empty"
    output_dir = f"./models/sft-{model_size}"
    print(f"[INFO] Output directory: {output_dir}")
    
    training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=hyper_parameters['num_train_epochs'],
            per_device_train_batch_size=hyper_parameters['per_device_train_batch_size'],
            per_device_eval_batch_size=hyper_parameters['per_device_eval_batch_size'],
            gradient_accumulation_steps=hyper_parameters['gradient_accumulation_steps'],
            save_steps=hyper_parameters['save_steps'],
            save_total_limit=hyper_parameters['save_total_limit'],
            eval_strategy=hyper_parameters['eval_strategy'],
            eval_steps=hyper_parameters['eval_steps'],
            save_strategy=hyper_parameters['save_strategy'],
            logging_dir="/var/logs",
            logging_steps=hyper_parameters['logging_steps'],
            overwrite_output_dir=True,
            fp16=hyper_parameters["fp16"],
            gradient_checkpointing=False,
            logging_first_step=True,
            weight_decay=0.01,  # Add regularization
            max_grad_norm=1.0,  # Add gradient clipping
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
            )
    
    # Train the model
    model.config.use_cache = False
    print("[INFO] Starting training")
    trainer.train()

    # Explicitly save the final model and tokenizer to the output directory
    print(f"[INFO] Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Optionally, save the trainer state
    trainer_state = {
        "best_model_checkpoint": output_dir,
        "best_metric": trainer.state.best_metric if hasattr(trainer.state, "best_metric") else None,
        "epoch": trainer.state.epoch,
        "global_step": trainer.state.global_step,
    }
    
    with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f)

    return model, trainer, test_dataset


def peft(model_size: str) -> Tuple[T5ForConditionalGeneration, Trainer, QueryReformulationDataset]:
    device, tokenizer, model = init_models(model_size, quantized=True)
    assert device == 'cuda', "PEFT requires a CUDA device"

    lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q", "k", "v"],
            lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
        
    hyper_parameters = get_optimised_hyperparameters()
    training_dataset, eval_dataset, test_dataset = create_datasets(tokenizer, hyper_parameters['sample_size'])

    output_dir = f"./models/peft-{model_size}"
    
    training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=hyper_parameters['num_train_epochs'],
            per_device_train_batch_size=hyper_parameters['per_device_train_batch_size'],
            per_device_eval_batch_size=hyper_parameters['per_device_eval_batch_size'],
            gradient_accumulation_steps=hyper_parameters['gradient_accumulation_steps'],
            save_steps=hyper_parameters['save_steps'],
            save_total_limit=hyper_parameters['save_total_limit'],
            eval_strategy=hyper_parameters['eval_strategy'],
            eval_steps=hyper_parameters['eval_steps'],
            save_strategy=hyper_parameters['save_strategy'],
            logging_dir="/var/logs",
            logging_steps=hyper_parameters['logging_steps'],
            overwrite_output_dir=True,
            fp16=hyper_parameters["fp16"],
            gradient_checkpointing=False,
            logging_first_step=True,
            weight_decay=0.01,
            max_grad_norm=1.0,
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
            )
    
    # Train the model
    model.config.use_cache = False
    print("[INFO] Starting training")
    trainer.train()

    # Explicitly save the final model and tokenizer to the output directory
    print(f"[INFO] Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Optionally, save the trainer state
    trainer_state = {
        "best_model_checkpoint": output_dir,
        "best_metric": trainer.state.best_metric if hasattr(trainer.state, "best_metric") else None,
        "epoch": trainer.state.epoch,
        "global_step": trainer.state.global_step,
    }
    
    with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f)

    return model, trainer, test_dataset



def benchmark(model: T5ForConditionalGeneration, trainer: Trainer, test_dataset: QueryReformulationDataset) -> None:
    # Evaluate the model against the test set
    print("[INFO] Evaluating model on test set...")
    model.config.use_cache = True
    test_results = trainer.evaluate(test_dataset)
    
    # Print test results as a markdown table
    print("\n### Test Results")
    print("| Metric | Value |")
    print("|--------|-------|")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"| {key} | {value:.4f} |")
        else:
            print(f"| {key} | {value} |")


def print_trainable_parameters(model: T5ForConditionalGeneration) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}",
    )


@click.command()
@click.option('--model-size', type=click.Choice(['small', 'base', 'large']), default='small',
              help='Size of the T5 model to use')
def main(model_size: str) -> None:
    """Train a query reformulation model using the specified parameters."""
    model, trainer, dataset = sft(model_size)
    #model, trainer, dataset = peft(model_size)
    benchmark(model, trainer, dataset)


if __name__ == "__main__":
    main()
