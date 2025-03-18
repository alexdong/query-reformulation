import sys
import os
import json
import torch
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration

from data import QueryReformulationDataset
from utils.init_models import init_models
device, tokenizer, model = init_models()
dataset = QueryReformulationDataset(tokenizer, dataset="full" if device == "cuda" else "dev")


if __name__ == "__main__":
    dataset = QueryReformulationDataset(tokenizer, dataset="dev")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(sample)
    print(sample["query_tokens"])
    print(sample["subquery_tokens"])
    print("Done.")
