import sys
import os
import json
import torch
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration

from data import QueryReformulationDataset
from utils.init_models import init_models

def fine_tune(model_size="base", dataset="full", training_epochs=1):
    device, tokenizer, model = init_models(model_size, use_sft_model=False)
    dataset = QueryReformulationDataset(tokenizer, dataset=dataset)

    training_args = TrainingArguments(
            output_dir=f"./models/sft-{model_size}",
            num_train_epochs=training_epochs,
            per_device_train_batch_size=8,
            save_steps=1_000,
            save_total_limit=2,
            logging_dir="/var/logs",
            logging_steps=1_000,
            overwrite_output_dir=True,
            )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    fine_tune(model_size="small", dataset="full", training_epochs=1)
    fine_tune(model_size="base", dataset="full", training_epochs=3)
