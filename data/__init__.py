import json
from pathlib import Path
from typing import Any, Dict, List, Literal

from datasets import Dataset
from transformers import T5Tokenizer


def load_dataset_from_jsonl(
    file_path: Path,
    split_role: Literal["train", "eval", "test"] = "train",
) -> List[Dict[str, str]]:
    """Load data from jsonl file and return as a list of dictionaries."""

    # The ratios are set to 85% training, 10% evaluation, and 5% testing.
    # 80-10-10 split won't work in local dev because MPS backend requires more RAM than I have.
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    if split_role == "train":
        return dataset[:int(0.85 * len(dataset))]
    elif split_role == "eval":
        return dataset[int(0.85 * len(dataset)):int(0.98 * len(dataset))]
    elif split_role == "test":
        return dataset[int(0.98 * len(dataset)):]


class QueryReformulationDataset:
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        dataset: str = "full",
        split_role: str = "train",
    ) -> None:
        self.tokenizer = tokenizer
        # shuffle the data please, ai!
        data = load_dataset_from_jsonl(Path(f"data/{dataset}.jsonl"), split_role=split_role)
        
        # Convert to HF Dataset format
        self.dataset = Dataset.from_dict({
            "input": [item.get("query") for item in data],
            "output": [item.get("subqueries") for item in data],
        })
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        input_text = f"reformulate: {item['input']}"
        output_text = item['output']
        
        input_tokens = self.tokenizer.encode_plus(
            input_text,
            max_length=64,  # Query Max tokens is 57 with P99 at 36
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        output_tokens = self.tokenizer.encode_plus(
            output_text,
            max_length=80,  # Expansion's P99 is 70.
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": input_tokens["input_ids"].squeeze(),
            "attention_mask": input_tokens["attention_mask"].squeeze(),
            "labels": output_tokens["input_ids"].squeeze(),
            "decoder_attention_mask": output_tokens["attention_mask"].squeeze(),
            # Keep original text for reference
            "input": input_text,
            "output": output_text,
        }


if __name__ == "__main__":
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = QueryReformulationDataset(tokenizer, dataset="dev", split_role="eval")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(sample.keys())
    print(sample["input"])
    print(sample["output"])
