import json
import os

import click
from transformers import (
    Trainer,
    TrainingArguments,
)

from benchmark.metric import compute_metrics
from data import QueryReformulationDataset
from utils.init_models import init_models


def fine_tune(model_size: str, dataset: str, training_epochs: int, batch_size: int = 8) -> None:
    device, tokenizer, model = init_models(model_size, use_sft_model=False)
    train_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="train")
    eval_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="eval")
    test_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="test")

    output_dir = f"./models/sft-{model_size}"
    
    training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=1_000,
            save_total_limit=2,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="/var/logs",
            logging_steps=1_000,
            overwrite_output_dir=True,
            fp16=device == "cuda",
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda x: compute_metrics(x, tokenizer),
            )
    
    # Train the model
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

    # Evaluate the model against the test set
    print(f"[INFO] Evaluating model on test set...")
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
@click.option('--dataset', type=str, default='dev',
              help='Dataset to use for training (dev or full)')
@click.option('--epochs', type=int, default=1,
              help='Number of training epochs')
@click.option('--batch-size', type=int, default=8,
              help='Batch size for training')
def main(model_size: str, dataset: str, epochs: int, batch_size: int) -> None:
    """Train a query reformulation model using the specified parameters."""
    print(f"[INFO] Training with model_size={model_size}, dataset={dataset}, epochs={epochs}, batch_size={batch_size}")
    fine_tune(model_size, dataset, epochs, batch_size)


if __name__ == "__main__":
    main()
