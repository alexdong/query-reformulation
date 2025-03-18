"""
Benchmark different model implementations for query reformulation.
"""
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple

import torch
from tqdm import tqdm
import onnxruntime as ort
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel
)

# Import utilities for different model types
from models import load_model as load_t5_model, generate_reformulation as generate_t5_reformulation, MODEL_SIZES as T5_MODEL_SIZES
from onnx_utils import export_to_onnx, generate_reformulation_onnx
from torchscript_utils import script_model, generate_reformulation_torchscript

# Define constants
DEV_DATASET = Path("datasets/dev.jsonl")
BERT_MODEL_SIZES = ["base", "large"]
MODEL_TYPES = ["t5", "bert"]

def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from a jsonl file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_bert_model(model_size: str, force_cpu: bool = False) -> Tuple[PreTrainedModel, AutoTokenizer, torch.device]:
    """Load a BERT model and tokenizer of specified size.
    
    Args:
        model_size: Size of the model ('base' or 'large')
        force_cpu: If True, forces the model to run on CPU even if GPU/MPS is available
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    assert model_size in BERT_MODEL_SIZES, f"Invalid model size: {model_size}"
    model_name = f"bert-{model_size}-uncased"
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
    # For query reformulation, we'll use sequence classification as a starting point
    # This can be adapted based on your specific approach
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)

    return model, tokenizer, device

def generate_bert_reformulation(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    query: str,
    device: torch.device,
) -> str:
    """Generate query reformulation using a BERT model.
    
    Note: This is a placeholder implementation. You'll need to adapt this
    based on how you fine-tune BERT for query reformulation.
    
    Args:
        model: BERT model
        tokenizer: The tokenizer for the model
        query: The query to reformulate
        device: The device to run inference on
        
    Returns:
        The reformulated query
    """
    # This is a simplified placeholder implementation
    # In practice, you would implement your specific approach for BERT-based reformulation
    input_text = f"reformulate: {query}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # This is just a placeholder - you would implement your specific logic here
    # For example, if you're using a token classification approach, you might:
    # 1. Get the token-level predictions
    # 2. Select tokens based on some criteria
    # 3. Combine them to form the reformulated query
    
    # For now, we'll just return the original query as a placeholder
    return f"BERT reformulation placeholder for: {query}"

def benchmark_model(
    model_type: str,
    model_size: str, 
    dataset: List[Dict[str, Any]], 
    force_cpu: bool = False,
    use_onnx: bool = False,
    use_torchscript: bool = False
) -> Dict[str, float]:
    """Benchmark the model on the dataset.
    
    Args:
        model_type: Type of model ('t5' or 'bert')
        model_size: Size of the model (depends on model_type)
        dataset: The dataset to benchmark on
        force_cpu: Whether to force CPU usage
        use_onnx: Whether to use ONNX runtime
        use_torchscript: Whether to use TorchScript
        
    Returns:
        Dictionary of benchmark statistics
    """
    # Load appropriate model based on model_type
    if model_type == "t5":
        original_model, tokenizer, device = load_t5_model(model_size, force_cpu)
        generate_fn = generate_t5_reformulation
    elif model_type == "bert":
        original_model, tokenizer, device = load_bert_model(model_size, force_cpu)
        generate_fn = generate_bert_reformulation
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = original_model  # Keep a reference to the original model
    runtime = "pytorch"
    
    # Handle TorchScript if requested
    if use_torchscript:
        try:
            scripted_model, script_path = script_model(original_model, tokenizer, model_size)
            model = scripted_model  # Use the scripted model
            runtime = "torchscript"
            print(f"[INFO] Using TorchScript model from {script_path}")
        except Exception as e:
            print(f"[ERROR] Failed to use TorchScript: {e}")
            print("[INFO] Falling back to PyTorch")
            use_torchscript = False
    # Handle ONNX if requested (and TorchScript not used)
    elif use_onnx:
        try:
            # Note: You would need to adapt export_to_onnx for BERT models
            onnx_path = export_to_onnx(original_model, tokenizer, model_size, model_type=model_type)
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
        if use_torchscript and model_type == "t5":
            # TorchScript function returns (reformulated_query, inference_time)
            _, _ = generate_reformulation_torchscript(
                model, tokenizer, dataset[0]["query"], device, original_model
            )
        elif use_onnx and model_type == "t5":
            # ONNX function just returns the reformulated query
            _ = generate_reformulation_onnx(model, tokenizer, dataset[0]["query"])
        else:
            # PyTorch function just returns the reformulated query
            _ = generate_fn(model, tokenizer, dataset[0]["query"], device)

    print(f"[INFO] Benchmarking {model_type}-{model_size} on {total_queries} queries...")
    print(f"[INFO] Using {runtime.upper()}")
    
    for item in tqdm(dataset, desc=f"Processing queries", unit="query"):
        query = item["query"]
        
        if use_torchscript and model_type == "t5":
            # TorchScript function already measures time internally
            reformulated_query, inference_time_ms = generate_reformulation_torchscript(
                model, tokenizer, query, device, original_model
            )
            # Convert ms to seconds for consistency
            query_time = inference_time_ms / 1000.0
        elif use_onnx and model_type == "t5":
            # For ONNX, we need to measure time ourselves
            query_start = time.time()
            reformulated_query = generate_reformulation_onnx(model, tokenizer, query)
            query_time = time.time() - query_start
        else:
            # For PyTorch, we need to measure time ourselves
            query_start = time.time()
            reformulated_query = generate_fn(model, tokenizer, query, device)
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
        "model_type": model_type,
        "model_size": model_size,
        "runtime": runtime,
        "average_time": total_time / total_queries if total_queries > 0 else 0,
        "median_time": median_time,
        "stddev_time": stddev_time,
        "p90_time": p90_time,
        "p95_time": p95_time,
        "p99_time": p99_time,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark model performance for query reformulation")
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="t5", 
        choices=MODEL_TYPES,
        help="Type of model to benchmark (t5 or bert)"
    )
    parser.add_argument(
        "--model-size", 
        type=str, 
        default="base",
        help="Size of the model (small/base/large for T5, base/large for BERT)"
    )
    parser.add_argument(
        "--force-cpu", 
        action="store_true", 
        help="Force CPU usage even if GPU is available"
    )
    parser.add_argument(
        "--use-onnx", 
        action="store_true", 
        help="Use ONNX runtime for inference"
    )
    parser.add_argument(
        "--use-torchscript", 
        action="store_true", 
        help="Use TorchScript for inference"
    )
    
    args = parser.parse_args()
    
    # Validate model size based on model type
    if args.model_type == "t5" and args.model_size not in T5_MODEL_SIZES:
        raise ValueError(f"Invalid model size for T5: {args.model_size}. Choose from {T5_MODEL_SIZES}")
    elif args.model_type == "bert" and args.model_size not in BERT_MODEL_SIZES:
        raise ValueError(f"Invalid model size for BERT: {args.model_size}. Choose from {BERT_MODEL_SIZES}")
    
    dataset = load_dataset(DEV_DATASET)
    print(f"[INFO] Loaded {len(dataset)} examples from {DEV_DATASET}")

    stats = benchmark_model(
        args.model_type,
        args.model_size, 
        dataset, 
        args.force_cpu, 
        args.use_onnx, 
        args.use_torchscript
    )
    
    # Pretty print stats
    print(f"\n{'=' * 50}")
    print(f"📊 RESULTS FOR {stats['model_type'].upper()}-{stats['model_size'].upper()} ({stats['runtime'].upper()}) 📊")
    print(f"{'=' * 50}")
    print(f"🕒 Average time per query: {stats['average_time']*1000:.2f} ms")
    print(f"🕒 Median time per query:  {stats['median_time']*1000:.2f} ms")
    print(f"📏 Standard deviation:     {stats['stddev_time']*1000:.2f} ms")
    print(f"📈 90th percentile (P90):  {stats['p90_time']*1000:.2f} ms")
    print(f"📈 95th percentile (P95):  {stats['p95_time']*1000:.2f} ms")
    print(f"📈 99th percentile (P99):  {stats['p99_time']*1000:.2f} ms")
    print(f"{'=' * 50}")
