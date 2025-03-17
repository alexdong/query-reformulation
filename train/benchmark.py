"""
Benchmark different model implementations for query reformulation.
"""
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from tqdm import tqdm
import onnxruntime as ort

from models import load_model, generate_reformulation, MODEL_SIZES
from onnx_utils import export_to_onnx, generate_reformulation_onnx
from torchscript_utils import save_as_torchscript, generate_reformulation_torchscript

DEV_DATASET = Path("datasets/dev.jsonl")

def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from a jsonl file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def benchmark_model(
    model_size: str, 
    dataset: List[Dict[str, Any]], 
    force_cpu: bool = False,
    use_onnx: bool = False,
    use_torchscript: bool = False
) -> Dict[str, float]:
    """Benchmark the model on the dataset.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        dataset: The dataset to benchmark on
        force_cpu: Whether to force CPU usage
        use_onnx: Whether to use ONNX runtime
        use_torchscript: Whether to use TorchScript
        
    Returns:
        Dictionary of benchmark statistics
    """
    # Load regular PyTorch model first
    model, tokenizer, device = load_model(model_size, force_cpu)
    
    runtime = "pytorch"
    
    # Handle TorchScript if requested
    if use_torchscript:
        try:
            script_path = save_as_torchscript(model, tokenizer, model_size)
            # Load the TorchScript model
            model = torch.jit.load(str(script_path), map_location=device)
            runtime = "torchscript"
            print(f"[INFO] Using TorchScript model from {script_path}")
        except Exception as e:
            print(f"[ERROR] Failed to use TorchScript: {e}")
            print("[INFO] Falling back to PyTorch")
            use_torchscript = False
    # Handle ONNX if requested (and TorchScript not used)
    elif use_onnx:
        try:
            onnx_path = export_to_onnx(model, tokenizer, model_size)
            if onnx_path:
                # Configure session options for better performance
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 4  # Adjust based on your CPU
                
                # Create inference session
                model = ort.InferenceSession(str(onnx_path), sess_options)
                runtime = "onnx"
                print(f"[INFO] Using ONNX Runtime on CPU")
            else:
                print("[INFO] ONNX export failed - using PyTorch instead")
        except Exception as e:
            print(f"[ERROR] Failed to use ONNX: {e}")
            print("[INFO] Falling back to PyTorch")
            use_onnx = False
    
    print(f"[INFO] Execute on {'CPU' if force_cpu or device.type == 'cpu' else device.type.upper()}")

    total_time = 0
    total_queries = len(dataset)
    query_times = []  # Track individual query times for statistics

    # Warm-up run
    if total_queries > 0:
        if use_torchscript:
            generate_reformulation_torchscript(model, tokenizer, dataset[0]["query"], device)
        elif use_onnx:
            generate_reformulation_onnx(model, tokenizer, dataset[0]["query"])
        else:
            generate_reformulation(model, tokenizer, dataset[0]["query"], device)

    print(f"[INFO] Benchmarking flan-t5-{model_size} on {total_queries} queries...")
    print(f"[INFO] Using {runtime.upper()}")
    
    for item in tqdm(dataset, desc=f"Processing queries", unit="query"):
        query = item["query"]
        query_start = time.time()
        
        if use_torchscript:
            reformulation = generate_reformulation_torchscript(model, tokenizer, query, device)
        elif use_onnx:
            reformulation = generate_reformulation_onnx(model, tokenizer, query)
        else:
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
        "runtime": "pytorch",
        "average_time": total_time / total_queries if total_queries > 0 else 0,
        "median_time": median_time,
        "stddev_time": stddev_time,
        "p90_time": p90_time,
        "p95_time": p95_time,
        "p99_time": p99_time,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Flan-T5 models for query reformulation")
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="base", 
                        help="Size of the model (small, base, large)")
    parser.add_argument("--force-cpu", action="store_true", 
                        help="Force using CPU even if GPU/MPS is available")
    parser.add_argument("--use-onnx", action="store_true",
                        help="Use ONNX Runtime for inference")
    parser.add_argument("--use-torchscript", action="store_true",
                        help="Use TorchScript for inference")
    
    args = parser.parse_args()
    
    # Load the dataset
    dataset = load_dataset(DEV_DATASET)
    print(f"[INFO] Loaded {len(dataset)} examples from {DEV_DATASET}")

    stats = benchmark_model(
        args.model_size, 
        dataset, 
        args.force_cpu, 
        args.use_onnx, 
        args.use_torchscript
    )
    
    # Pretty print stats
    print(f"\n{'=' * 50}")
    print(f"📊 RESULTS FOR FLAN-T5-{stats['model_size'].upper()} ({stats['runtime'].upper()}) 📊")
    print(f"{'=' * 50}")
    print(f"🕒 Average time per query: {stats['average_time']*1000:.2f} ms")
    print(f"🕒 Median time per query:  {stats['median_time']*1000:.2f} ms")
    print(f"📏 Standard deviation:     {stats['stddev_time']*1000:.2f} ms")
    print(f"📈 90th percentile (P90):  {stats['p90_time']*1000:.2f} ms")
    print(f"📈 95th percentile (P95):  {stats['p95_time']*1000:.2f} ms")
    print(f"📈 99th percentile (P99):  {stats['p99_time']*1000:.2f} ms")
    print(f"{'=' * 50}")
