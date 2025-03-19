import json
import os

import click
from transformers import (
    Trainer,
    TrainingArguments,
)

from benchmark.metric import compute_metrics
from data import create_datasets
from train.params import get_optimised_hyperparameters
from utils.init_models import init_models


def sft(model_size: str) -> Tuple[T5ForConditionalGeneration, Trainer, QueryReformulationDataset]:
    device, tokenizer, model = init_models(model_size, use_sft_model=False)
    hyper_parameters = get_optimised_hyperparameters()
    training_dataset, eval_dataset, test_dataset = create_datasets(tokenizer, hyper_parameters['sample_size'])

    output_dir = f"./models/sft-{model_size}"
    
    training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=hyper_parameters['num_train_epochs'],
            per_device_train_batch_size=hyper_parameters['per_device_train_batch_size'],
            save_steps=hyper_parameters['save_steps'],
            save_total_limit=hyper_parameters['save_total_limit'],
            eval_strategy=hyper_parameters['eval_strategy'],
            save_strategy=hyper_parameters['save_strategy'],
            logging_dir="/var/logs",
            logging_steps=hyper_parameters['logging_steps'],
            overwrite_output_dir=True,
            fp16=hyper_parameters['fp16'],
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda x: compute_metrics(x, tokenizer),
            )
    
    # Train the model
    model.config.use_cache = False
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
    device, tokenizer, model = init_models(model_size, use_sft_model=True)
    if device == 'cuda':
        pass

    hyper_parameters = get_optimised_hyperparameters()
    training_dataset, eval_dataset, test_dataset = create_datasets(tokenizer, hyper_parameters['sample_size'])

    output_dir = f"./models/peft-{model_size}"
    
    training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=hyper_parameters['num_train_epochs'],
            per_device_train_batch_size=hyper_parameters['per_device_train_batch_size'],
            save_steps=hyper_parameters['save_steps'],
            save_total_limit=hyper_parameters['save_total_limit'],
            eval_strategy=hyper_parameters['eval_strategy'],
            save_strategy=hyper_parameters['save_strategy'],
            logging_dir="/var/logs",
            logging_steps=hyper_parameters['logging_steps'],
            overwrite_output_dir=True,
            fp16=hyper_parameters['fp16'],
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda x: compute_metrics(x, tokenizer),
            )
    
    # Train the model
    model.config.use_cache = False
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
    
    print(f"[INFO] Training complete. Final model saved to {output_dir}")




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
