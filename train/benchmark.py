import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
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

def load_model(model_size: str, force_cpu: bool = False) -> tuple:
    """Load the Flan-T5 model and tokenizer of specified size.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: If True, forces the model to run on CPU even if GPU/MPS is available
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    model_name = f"google/flan-t5-{model_size}"
    print(f"[INFO] Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Determine device
    if force_cpu:
        device = torch.device("cpu")
        print("[INFO] Forcing CPU usage as requested")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INFO] Using CUDA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU (no GPU/MPS available)")
    
    # Load model to the selected device
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)

    return model, tokenizer, device

def generate_reformulation(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    query: str,
    device: torch.device,
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

def benchmark_model(model_size: str, dataset: List[Dict[str, Any]], force_cpu: bool = False) -> Dict[str, float]:
    """Benchmark the model on the dataset."""
    model, tokenizer, device = load_model(model_size, force_cpu)

    total_time = 0
    total_queries = len(dataset)

    # Warm-up run
    if total_queries > 0:
        generate_reformulation(model, tokenizer, dataset[0]["query"], device)

    print(f"[INFO] Benchmarking flan-t5-{model_size} on {total_queries} queries...")
    start_time = time.time()

    for item in tqdm(dataset, desc=f"Processing queries", unit="query"):
        query = item["query"]
        query_start = time.time()
        reformulation = generate_reformulation(model, tokenizer, query, device)
        query_time = time.time() - query_start
        total_time += query_time

        """
        print(f"Query: {query}")
        print(f"Reformulation: {reformulation}")
        print(f"Time: {query_time:.4f}s")
        print("-" * 50)
        """

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
        result = benchmark_model(model_size, dataset, force_cpu=True)
        results.append(result)

        print(f"\n[RESULTS] flan-t5-{model_size}:")
        print(f"Average time per query: {result['average_time']:.4f}s")
        # add median time per query and stddev, ai!
        print("=" * 50)

    return results

if __name__ == "__main__":
    for model_size in MODEL_SIZES:
        run_benchmarks(args.model_size)
