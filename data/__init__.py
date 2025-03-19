import json
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from datasets import Dataset
from transformers import T5Tokenizer


class QueryReformulationDataset:
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        data: List[Dict[str, str]],
    ) -> None:
        self.tokenizer = tokenizer
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


def create_datasets(tokenizer: T5Tokenizer, sample_size: float) -> Tuple[QueryReformulationDataset]:
    dataset = []
    with open(Path("./data/full.jsonl"), "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    random.shuffle(dataset)
    dataset = random.sample(dataset, int(len(dataset) * sample_size))

    def _create(role: Literal["train", "eval", "test"]) -> QueryReformulationDataset:
        if role == "train":
            data = dataset[:int(0.8 * len(dataset))]
        elif role == "eval":
            data = dataset[int(0.8 * len(dataset)):int(0.9 * len(dataset))]
        else:
            data = dataset[int(0.9 * len(dataset)):]
        return QueryReformulationDataset(tokenizer, data)

    train_dataset = _create("train")
    eval_dataset = _create("eval")
    test_dataset = _create("test")
    return train_dataset, eval_dataset, test_dataset

if __name__ == "__main__":
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = create_datasets(tokenizer, 0.1)
    print(dataset)
