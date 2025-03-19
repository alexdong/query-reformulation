import sys
import os
import json
import torch
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from bert_score import score
import click

from data import QueryReformulationDataset
from utils.init_models import init_models


def compute_metrics(eval_pred):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    predictions, labels = eval_pred
    predictions = [[t if t != -100 else tokenizer.pad_token_id for t in p] for p in predictions]
    labels = [[t if t != -100 else tokenizer.pad_token_id for t in l] for l in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    P, R, F1 = score(decoded_preds, decoded_labels, lang="en", model_type="bert-base-uncased", device="cuda" if torch.cuda.is_available() else "cpu")
    return {"bertscore_f1": F1.mean().item()}


def fine_tune(model_size="base", dataset="full", training_epochs=1):
    device, tokenizer, model = init_models(model_size, use_sft_model=False)
    train_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="train")
    eval_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="eval")

    training_args = TrainingArguments(
            output_dir=f"./models/sft-{model_size}",
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
            compute_metrics=compute_metrics,
            )
    trainer.train()


def evaluate(model_size="base", dataset="full"):
    device, tokenizer, model = init_models(model_size, use_sft_model=True)
    eval_dataset = QueryReformulationDataset(tokenizer, dataset=dataset, split_role="eval")
    
    training_args = TrainingArguments(
            output_dir=f"./models/eval-{model_size}",
            per_device_eval_batch_size=8,
            logging_dir="/var/logs",
            )
    
    trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            )
    
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
    fine_tune(model_size=model_size, dataset=dataset, training_epochs=epochs)


@main.command()
@click.option('--model-size', type=click.Choice(['small', 'base', 'large']), default='small',
              help='Size of the T5 model to use')
@click.option('--dataset', type=str, default='dev',
              help='Dataset to use for evaluation (dev or full)')
def eval(model_size, dataset):
    """Evaluate a trained query reformulation model using the specified parameters."""
    print(f"[INFO] Evaluating with model_size={model_size}, dataset={dataset}")
    evaluate(model_size=model_size, dataset=dataset)


if __name__ == "__main__":
    main()
