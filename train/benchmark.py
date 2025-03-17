import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Add these imports for ONNX support
import onnx
import onnxruntime as ort
import numpy as np

DEV_DATASET = Path("datasets/dev.jsonl")
MODEL_SIZES = ["small", "base", "large"]
ONNX_DIR = Path("models/onnx")

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

def quantize_and_export_to_onnx(
    model: AutoModelForSeq2SeqLM, 
    tokenizer: AutoTokenizer, 
    model_size: str
) -> Path:
    """Quantize the model and export to ONNX format.
    
    Args:
        model: The PyTorch model to quantize and export
        tokenizer: The tokenizer for the model
        model_size: Size of the model ('small', 'base', or 'large')
        
    Returns:
        Path to the exported ONNX model
    """
    print(f"[INFO] Quantizing and exporting model to ONNX...")
    
    # Create directory if it doesn't exist
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_DIR / f"flan-t5-{model_size}-quantized.onnx"
    
    # Skip if model already exists
    if onnx_path.exists():
        print(f"[INFO] ONNX model already exists at {onnx_path}")
        return onnx_path
    
    # Prepare model for export
    model.eval()
    
    # Quantize the model to int8
    # Note: We're using dynamic quantization which is applied during export
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Create dummy input for tracing
    dummy_input = tokenizer("reformulate:This is a test query", return_tensors="pt")
    
    # Export the model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            quantized_model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            onnx_path,
            opset_version=12,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
        )
    
    print(f"[INFO] Model exported to {onnx_path}")
    return onnx_path

def generate_reformulation(
    model: Union[AutoModelForSeq2SeqLM, ort.InferenceSession],
    tokenizer: AutoTokenizer,
    query: str,
    device: torch.device,
    use_onnx: bool = False
) -> str:
    """Generate query reformulation using the model.
    
    Args:
        model: Either a PyTorch model or ONNX InferenceSession
        tokenizer: The tokenizer for the model
        query: The query to reformulate
        device: The device to run inference on
        use_onnx: Whether to use ONNX runtime for inference
        
    Returns:
        The reformulated query
    """
    input_text = f"reformulate:{query}"
    
    if use_onnx:
        # ONNX Runtime inference
        inputs = tokenizer(input_text, return_tensors="pt")
        ort_inputs = {
            "input_ids": inputs["input_ids"].cpu().numpy(),
            "attention_mask": inputs["attention_mask"].cpu().numpy()
        }
        
        # Run inference
        ort_outputs = model.run(None, ort_inputs)
        
        # Process outputs
        # Note: This is simplified and may need adjustment based on the actual model output
        output_ids = np.argmax(ort_outputs[0], axis=-1)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        # PyTorch inference
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_return_sequences=1,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def benchmark_model(
    model_size: str, 
    dataset: List[Dict[str, Any]], 
    force_cpu: bool = False,
    use_onnx: bool = False
) -> Dict[str, float]:
    """Benchmark the model on the dataset.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        dataset: The dataset to benchmark on
        force_cpu: Whether to force CPU usage
        use_onnx: Whether to use ONNX runtime
        
    Returns:
        Dictionary of benchmark statistics
    """
    if use_onnx:
        # Load PyTorch model first to export to ONNX if needed
        model, tokenizer, device = load_model(model_size, force_cpu=True)  # Force CPU for ONNX export
        
        # Check if ONNX model already exists
        onnx_path = ONNX_DIR / f"flan-t5-{model_size}-quantized.onnx"
        if not onnx_path.exists():
            # Quantize and export to ONNX only if it doesn't exist
            onnx_path = quantize_and_export_to_onnx(model, tokenizer, model_size)
        else:
            print(f"[INFO] Using existing ONNX model at {onnx_path}")
        
        # Create ONNX Runtime session
        print(f"[INFO] Creating ONNX Runtime session...")
        # Configure session options for better performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Adjust based on your CPU
        
        # Create inference session
        model = ort.InferenceSession(str(onnx_path), sess_options)
        device = torch.device("cpu")  # ONNX Runtime uses CPU
        print(f"[INFO] Using ONNX Runtime on CPU")
    else:
        # Load regular PyTorch model
        model, tokenizer, device = load_model(model_size, force_cpu)
        print(f"[INFO] Execute on CPU")

    total_time = 0
    total_queries = len(dataset)
    query_times = []  # Track individual query times for statistics

    # Warm-up run
    generate_reformulation(model, tokenizer, dataset[0]["query"], device, use_onnx)

    print(f"[INFO] Benchmarking flan-t5-{model_size} on {total_queries} queries...")
    print(f"[INFO] Using {'ONNX Runtime' if use_onnx else 'PyTorch'}")
    
    for item in tqdm(dataset, desc=f"Processing queries", unit="query"):
        query = item["query"]
        query_start = time.time()
        reformulation = generate_reformulation(model, tokenizer, query, device, use_onnx)
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
        "runtime": "onnx" if use_onnx else "pytorch",
        "average_time": total_time / total_queries if total_queries > 0 else 0,
        "median_time": median_time,
        "stddev_time": stddev_time,
        "p90_time": p90_time,
        "p95_time": p95_time,
        "p99_time": p99_time,
    }


if __name__ == "__main__":
    model_size = "base"
    force_cpu = True
    use_onnx = True
    
    # Load the dataset
    dataset = load_dataset(DEV_DATASET)

    stats = benchmark_model(model_size, dataset, force_cpu, use_onnx)
    
    # Pretty print stats
    print(f"\n{'=' * 50}")
    print(f"📊 RESULTS FOR FLAN-T5-{model_size.upper()} ({stats['runtime'].upper()}) 📊")
    print(f"{'=' * 50}")
    print(f"🕒 Average time per query: {stats['average_time']*1000:.2f} ms")
    print(f"🕒 Median time per query:  {stats['median_time']*1000:.2f} ms")
    print(f"📏 Standard deviation:     {stats['stddev_time']*1000:.2f} ms")
    print(f"📈 90th percentile (P90):  {stats['p90_time']*1000:.2f} ms")
    print(f"📈 95th percentile (P95):  {stats['p95_time']*1000:.2f} ms")
    print(f"📈 99th percentile (P99):  {stats['p99_time']*1000:.2f} ms")
    print(f"{'=' * 50}")
