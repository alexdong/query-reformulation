import json
import statistics
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

def load_model(model_size: str, force_cpu: bool = False) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device]:
    """Load the Flan-T5 model and tokenizer of specified size.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: If True, forces the model to run on CPU even if GPU/MPS is available
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    assert model_size in MODEL_SIZES, f"Invalid model size: {model_size}"
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

# Quantise the model and run on onnxruntime, ai!
def benchmark_model(model_size: str, dataset: List[Dict[str, Any]], force_cpu: bool = False) -> Dict[str, float]:
    """Benchmark the model on the dataset."""
    model, tokenizer, device = load_model(model_size, force_cpu)

    total_time = 0
    total_queries = len(dataset)
    query_times = []  # Track individual query times for statistics

    generate_reformulation(model, tokenizer, dataset[0]["query"], device)

    print(f"[INFO] Benchmarking flan-t5-{model_size} on {total_queries} queries...")
    for item in tqdm(dataset, desc=f"Processing queries", unit="query"):
        query = item["query"]
        query_start = time.time()
        reformulation = generate_reformulation(model, tokenizer, query, device)
        query_time = time.time() - query_start
        total_time += query_time
        query_times.append(query_time)  # Store individual query time

    # Calculate statistics
    query_times.sort()  # Sort for percentile calculations
    median_time = statistics.median(query_times) if query_times else 0
    stddev_time = statistics.stdev(query_times) if len(query_times) > 1 else 0
    
    # Calculate percentiles
    p90_index = int(len(query_times) * 0.9)
    p95_index = int(len(query_times) * 0.95)
    p99_index = int(len(query_times) * 0.99)
    
    p90_time = query_times[p90_index] if query_times and p90_index < len(query_times) else 0
    p95_time = query_times[p95_index] if query_times and p95_index < len(query_times) else 0
    p99_time = query_times[p99_index] if query_times and p99_index < len(query_times) else 0

    return {
        "model_size": model_size,
        "average_time": total_time / total_queries if total_queries > 0 else 0,
        "median_time": median_time,
        "stddev_time": stddev_time,
        "p90_time": p90_time,
        "p95_time": p95_time,
        "p99_time": p99_time,
    }


if __name__ == "__main__":
    dataset = load_dataset(DEV_DATASET)
    print(f"[INFO] Loaded {len(dataset)} examples from {DEV_DATASET}")

    for model_size in MODEL_SIZES:
        stats = benchmark_model(model_size, dataset, force_cpu=True)
        
        # Pretty print stats
        print(f"\n{'=' * 50}")
        print(f"📊 RESULTS FOR FLAN-T5-{model_size.upper()} 📊")
        print(f"{'=' * 50}")
        print(f"🕒 Average time per query: {stats['average_time']*1000:.2f} ms")
        print(f"🕒 Median time per query:  {stats['median_time']*1000:.2f} ms")
        print(f"📏 Standard deviation:     {stats['stddev_time']*1000:.2f} ms")
        print(f"📈 90th percentile (P90):  {stats['p90_time']*1000:.2f} ms")
        print(f"📈 95th percentile (P95):  {stats['p95_time']*1000:.2f} ms")
        print(f"📈 99th percentile (P99):  {stats['p99_time']*1000:.2f} ms")
        print(f"{'=' * 50}")
