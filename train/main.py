from datasets import Dataset
import json
import torch
from transformers import T5Tokenizer

def get_backend_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
device = get_backend_device()

model_size = "base" if device == "cuda" else "small"
model_name = f"google/flan-t5-{model_size}"
training_dataset = "datasets/full.jsonl" if device == "cuda" else "datasets/dev.jsonl"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

class QueryReformulationDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = self.load_dataset(training_dataset)

    def load_dataset(self, file_path):
        dataset = []
        with open(file_path, "r") as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item.get("query")
        subqueries = item.get("subqueries")

        query_tokens = self.tokenizer.encode_plus(
                query,
                max_length=64, # Query Max tokens is 57 with P99 at 36
                padding="max_length",
                truncation=True,
                return_tensors="pt"
                )

        subquery_tokens = self.tokenizer.encode_plus(
                subqueries,
                max_length=80, # Expansion's P99 is 70.
                padding="max_length",
                truncation=True,
                return_tensors="pt"
                )

        return {
            "query": query,
            "query_tokens": query_tokens,
            "subqueries": subqueries,
            "subquery_tokens": subquery_tokens,
        }


if __name__ == "__main__":
    dataset = QueryReformulationDataset(tokenizer)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(sample)
    print(sample["query_tokens"])
    print(sample["subquery_tokens"])
    print("Done.")
