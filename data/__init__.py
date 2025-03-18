from datasets import Dataset
import json
from pathlib import Path

class QueryReformulationDataset:
    def __init__(self, tokenizer, dataset="full"):
        self.tokenizer = tokenizer
        self._data = self.load_dataset(Path(f"datasets/{dataset}.jsonl"))

    def load_dataset(self, file_path):
        dataset = []
        with open(file_path, "r") as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
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
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = QueryReformulationDataset(tokenizer, dataset="dev")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(sample)
    print(sample["query_tokens"])
    print(sample["subquery_tokens"])
    print("Done.")
