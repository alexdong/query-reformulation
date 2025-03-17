import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEV_DATASET = Path("datasets/dev.jsonl")
MODEL_SIZES = ["small", "base", "large"]

def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from a jsonl file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# TODO: add an option to run only on CPU.
def load_model(model_size: str) -> tuple:
    """Load the Flan-T5 model and tokenizer of specified size."""
    model_name = f"google/flan-t5-{model_size}"
    print(f"[INFO] Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, tokenizer, device

def generate_reformulation(
    model: AutoModelForSeq2SeqLM, 
    tokenizer: AutoTokenizer, 
    query: str, 
    device: torch.device
) -> str:
    """Generate query reformulation using the model."""
    input_text = f"reformulate:{query}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_return_sequences=1,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def benchmark_model(model_size: str, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
    """Benchmark the model on the dataset."""
    model, tokenizer, device = load_model(model_size)

    total_time = 0
    total_queries = len(dataset)

    # Warm-up run
    if total_queries > 0:
        generate_reformulation(model, tokenizer, dataset[0]["query"], device)

    print(f"[INFO] Benchmarking flan-t5-{model_size} on {total_queries} queries...")
    start_time = time.time()

    for item in dataset:
        query = item["query"]
        query_start = time.time()
        reformulation = generate_reformulation(model, tokenizer, query, device)
        query_time = time.time() - query_start
        total_time += query_time

        print(f"Query: {query}")
        print(f"Reformulation: {reformulation}")
        print(f"Time: {query_time:.4f}s")
        print("-" * 50)

    end_time = time.time()

    results = {
        "model_size": model_size,
        "total_time": end_time - start_time,
        "average_time": total_time / total_queries if total_queries > 0 else 0,
        "queries_per_second": total_queries / total_time if total_time > 0 else 0,
        "total_queries": total_queries,
    }

    return results

def run_benchmarks(model_sizes: List[str] = MODEL_SIZES) -> List[Dict[str, float]]:
    """Run benchmarks for all specified model sizes."""
    dataset = load_dataset(DEV_DATASET)

    print(f"[INFO] Loaded {len(dataset)} examples from {DEV_DATASET}")

    results = []
    for model_size in model_sizes:
        result = benchmark_model(model_size, dataset)
        results.append(result)

        print(f"\n[RESULTS] flan-t5-{model_size}:")
        print(f"Total time: {result['total_time']:.2f}s")
        print(f"Average time per query: {result['average_time']:.4f}s")
        print(f"Queries per second: {result['queries_per_second']:.2f}")
        print("=" * 50)

    return results

if __name__ == "__main__":
    run_benchmarks(MODEL_SIZES)
