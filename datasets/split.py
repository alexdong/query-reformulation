#!/usr/bin/env python3
"""
Split the full dataset into training, validation, test, and dev sets.

This script takes the full.jsonl dataset and splits it into:
- training.jsonl (80%)
- validation.jsonl (10%)
- test.jsonl (10%)
- dev.jsonl (200 examples from training)
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

# Define directories and files
DATASET_DIR = Path("datasets")
FULL_DATASET = DATASET_DIR / "full.jsonl"
TRAINING_DATASET = DATASET_DIR / "training.jsonl"
VALIDATION_DATASET = DATASET_DIR / "validation.jsonl"
TEST_DATASET = DATASET_DIR / "test.jsonl"
DEV_DATASET = DATASET_DIR / "dev.jsonl"

# Define split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DEV_SIZE = 200  # Number of examples for dev set


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from a JSONL file."""
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def save_dataset(dataset: List[Dict[str, Any]], file_path: Path) -> None:
    """Save dataset to a JSONL file."""
    with open(file_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")


def split_dataset() -> None:
    """Split the full dataset into training, validation, test, and dev sets."""
    if not FULL_DATASET.exists():
        print(f"[ERROR] Full dataset not found at {FULL_DATASET}")
        return
    
    print(f"[INFO] Loading full dataset from {FULL_DATASET}")
    dataset = load_dataset(FULL_DATASET)
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    
    # Split the dataset
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    # Create dev set (subset of training data)
    dev_data = train_data[:min(DEV_SIZE, len(train_data))]
    
    # Save the datasets
    save_dataset(train_data, TRAINING_DATASET)
    save_dataset(val_data, VALIDATION_DATASET)
    save_dataset(test_data, TEST_DATASET)
    save_dataset(dev_data, DEV_DATASET)
    
    print("[INFO] Dataset split complete:")
    print(f"  - Training: {len(train_data)} examples")
    print(f"  - Validation: {len(val_data)} examples")
    print(f"  - Test: {len(test_data)} examples")
    print(f"  - Dev: {len(dev_data)} examples")


if __name__ == "__main__":
    split_dataset()
