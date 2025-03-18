from datasets import Dataset
import json
from pathlib import Path

class QueryReformulationDataset(Dataset):
    def __init__(self, tokenizer, dataset="full"):
        self.tokenizer = tokenizer
        self._data = self.load_dataset(Path(f"data/{dataset}.jsonl"))

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
        input = f"reformulate: {item.get('query')}"  # add the instruction
        output = item.get("subqueries")

        input_tokens = self.tokenizer.encode_plus(
                input,
                max_length=64, # Query Max tokens is 57 with P99 at 36
                padding="max_length",
                truncation=True,
                return_tensors="pt"
                )

        output_tokens = self.tokenizer.encode_plus(
                output,
                max_length=80, # Expansion's P99 is 70.
                padding="max_length",
                truncation=True,
                return_tensors="pt"
                )

        return {
            "input": input,
            "input_tokens": input_tokens,
            "output": output,
            "output_tokens": output_tokens,
        }


if __name__ == "__main__":
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = QueryReformulationDataset(tokenizer, dataset="dev")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(sample)
    print(sample["input"])
    print(sample["output"])
