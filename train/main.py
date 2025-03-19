import sys
import os
import json
import torch
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from bert_score import score
import click
import numpy as np

from data import QueryReformulationDataset
from utils.init_models import init_models


def compute_metrics(eval_pred, tokenizer, model_size, device):
    predictions, labels = eval_pred

    # Process each sequence individually
    decoded_preds = []
    decoded_labels = []

    # Convert to numpy if they're tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()


    for pred, label in zip(predictions, labels):
        # Replace -100 with pad_token_id
        pred_processed = np.where(pred != -100, pred, tokenizer.pad_token_id).tolist()
        label_processed = np.where(label != -100, label, tokenizer.pad_token_id).tolist()

        # Decode to text
        decoded_preds.append(tokenizer.decode(pred_processed, skip_special_tokens=True))
        decoded_labels.append(tokenizer.decode(label_processed, skip_special_tokens=True))

    # Calculate BERTScore
    P, R, F1 = score(decoded_preds, decoded_labels, lang="en",
model_type="microsoft/roberta-large", device=device)
    return {"bertscore_f1": F1.mean().item()}


def fine_tune(model_size, dataset, training_epochs):
    device, tokenizer, model = init_models(model_size, use_sft_model=False)
    train_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="train")
    eval_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="eval")

    output_dir = f"./models/sft-{model_size}"
    
    training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_epochs,
            per_device_train_batch_size=8,
            save_steps=1_000,
            save_total_limit=2,
            eval_strategy="steps",
            save_strategy="epoch",
            logging_dir="/var/logs",
            logging_steps=1_000,
            overwrite_output_dir=True,
            )

    trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda x: compute_metrics(x, tokenizer, model_size, device)
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
        "global_step": trainer.state.global_step
    }
    
    with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f)
    
    print(f"[INFO] Training complete. Final model saved to {output_dir}")


def evaluate(model_size, dataset):
    device, tokenizer, model = init_models(model_size, use_sft_model=True)
    test_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="test")
    
    training_args = TrainingArguments(
            output_dir=f"./models/sft-{model_size}",
            per_device_eval_batch_size=8,
            logging_dir="/var/logs",
            logging_steps=10,
            )
    
    trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=lambda x: compute_metrics(x, tokenizer, model_size, device)
            )
    
    print(f"[INFO] Evaluating model on {dataset} dataset")
    results = trainer.evaluate()
    print(f"[INFO] Evaluation results: {results}")
    return results


@click.group()
def main():
    """Train or evaluate a query reformulation model."""
    pass


@main.command()
@click.option('--model-size', type=click.Choice(['small', 'base', 'large']), default='small', 
              help='Size of the T5 model to use')
@click.option('--dataset', type=str, default='dev', 
              help='Dataset to use for training (dev or full)')
@click.option('--epochs', type=int, default=1, 
              help='Number of training epochs')
def train(model_size, dataset, epochs):
    """Train a query reformulation model using the specified parameters."""
    print(f"[INFO] Training with model_size={model_size}, dataset={dataset}, epochs={epochs}")
    fine_tune(model_size, dataset, epochs)


@main.command()
@click.option('--model-size', type=click.Choice(['small', 'base', 'large']), default='small',
              help='Size of the T5 model to use')
@click.option('--dataset', type=str, default='dev',
              help='Dataset to use for evaluation (dev or full)')
def eval(model_size, dataset):
    """Evaluate a trained query reformulation model using the specified parameters."""
    print(f"[INFO] Evaluating with model_size={model_size}, dataset={dataset}")
    evaluate(model_size, dataset)


if __name__ == "__main__":
    main()
