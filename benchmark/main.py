"""
Benchmark different model implementations for query reformulation.
"""
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import onnxruntime as ort
import torch
from onnx_utils import export_to_onnx, generate_reformulation_onnx
from torchscript_utils import generate_reformulation_torchscript, script_model
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    ModernBertConfig,
    ModernBertModel,
    PreTrainedModel,
)

from models import MODEL_SIZES as T5_MODEL_SIZES
from models import generate_reformulation as generate_t5_reformulation

# Import utilities for different model types
from models import load_model as load_t5_model

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
    """Load a ModernBert model and tokenizer of specified size.
    
    Args:
        model_size: Size of the model ('base' or 'large')
        force_cpu: If True, forces the model to run on CPU even if GPU/MPS is available
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    assert model_size in BERT_MODEL_SIZES, f"Invalid model size: {model_size}"
    
    # Use standard BERT tokenizer but ModernBert model
    model_name = f"bert-{model_size}-uncased"
    print(f"[INFO] Loading ModernBert-{model_size}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Force CPU for ModernBert due to MPS compatibility issues
    device = torch.device("cpu")
    print("[INFO] Forcing CPU usage for ModernBert (MPS compatibility issues)")
    
    # Load ModernBert model with appropriate configuration
    config = ModernBertConfig.from_pretrained(model_name)
    model = ModernBertModel(config)
    
    # Add a classification head for query reformulation
    # This is a placeholder - you'll need to adapt based on your specific approach
    model.classifier = torch.nn.Linear(config.hidden_size, 2)  # Simple binary classifier as placeholder
    
    model = model.to(device)
    return model, tokenizer, device

def generate_bert_reformulation(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    query: str,
    device: torch.device,
) -> str:
    """Generate query reformulation using a ModernBert model.
    
    Note: This is a placeholder implementation. You'll need to adapt this
    based on how you fine-tune ModernBert for query reformulation.
    
    Args:
        model: ModernBert model
        tokenizer: The tokenizer for the model
        query: The query to reformulate
        device: The device to run inference on
        
    Returns:
        The reformulated query
    """
    # This is a simplified placeholder implementation
    input_text = f"reformulate: {query}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Remove token_type_ids which ModernBertModel doesn't accept
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
    
    with torch.no_grad():
        # Get the ModernBert embeddings
        outputs = model(**inputs)
        # Use the [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Pass through the classifier
        model.classifier(cls_output)
    
    # This is just a placeholder - you would implement your specific logic here
    # For a real implementation, you might:
    # 1. Use a more sophisticated output head
    # 2. Implement a token-level approach for reformulation
    # 3. Fine-tune the model on your query reformulation dataset
    
    # For now, we'll just return a placeholder
    return f"ModernBert reformulation for: {query}"

def benchmark_model(
    model_type: str,
    model_size: str,
    dataset: List[Dict[str, Any]],
    force_cpu: bool = False,
    use_onnx: bool = False,
    use_torchscript: bool = False,
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
            scripted_model, script_path = script_model(original_model, tokenizer, model_size, quantize=args.quantize)
            model = scripted_model  # Use the scripted model
            runtime = "torchscript" + ("_quantized" if args.quantize else "")
            print(f"[INFO] Using {'quantized ' if args.quantize else ''}TorchScript model from {script_path}")
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
                print("[INFO] Using ONNX Runtime on CPU")
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
                model, tokenizer, dataset[0]["query"], device, original_model,
            )
        elif use_onnx and model_type == "t5":
            # ONNX function just returns the reformulated query
            _ = generate_reformulation_onnx(model, tokenizer, dataset[0]["query"])
        else:
            # PyTorch function just returns the reformulated query
            _ = generate_fn(model, tokenizer, dataset[0]["query"], device)

    print(f"[INFO] Benchmarking {model_type}-{model_size} on {total_queries} queries...")
    print(f"[INFO] Using {runtime.upper()}")
    
    for item in tqdm(dataset, desc="Processing queries", unit="query"):
        query = item["query"]
        
        if use_torchscript and model_type == "t5":
            # TorchScript function already measures time internally
            reformulated_query, inference_time_ms = generate_reformulation_torchscript(
                model, tokenizer, query, device, original_model,
            )
            # Convert ms to seconds for consistency
            query_time = inference_time_ms / 1000.0
        elif use_onnx and model_type == "t5":
            # For ONNX, we need to measure time ourselves
            query_start = time.time()
            generate_reformulation_onnx(model, tokenizer, query)
            query_time = time.time() - query_start
        else:
            # For PyTorch, we need to measure time ourselves
            query_start = time.time()
            generate_fn(model, tokenizer, query, device)
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
        default="bert",
        choices=MODEL_TYPES,
        help="Type of model to benchmark (t5 or bert)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large",
        help="Size of the model (small/base/large for T5, base/large for BERT)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available (for T5 models, BERT always uses CPU)",
    )
    parser.add_argument(
        "--use-onnx",
        action="store_false",
        help="Use ONNX runtime for inference",
    )
    parser.add_argument(
        "--use-torchscript",
        action="store_true",
        help="Use TorchScript for inference",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use quantized model (only applies with --use-torchscript)",
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
        args.use_torchscript,
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
import os
import json
import time
from pathlib import Path
import torch
import click
from tqdm import tqdm
from typing import List, Dict, Any

from data import QueryReformulationDataset, load_dataset_from_jsonl
from models.query import QueryReformulator
from benchmark.metric import compute_metrics

DEV_DATASET = Path("datasets/dev.jsonl")
BERT_MODEL_SIZES = ["base", "large"]
MODEL_TYPES = ["t5", "bert"]

def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load a dataset from a JSONL file."""
    return load_dataset_from_jsonl(file_path)

def evaluate(model_size: str, dataset: str, batch_size: int = 8, verbose: bool = False):
    """Evaluate a trained query reformulation model using direct inference.
    
    Args:
        model_size: Size of the T5 model ('small', 'base', or 'large')
        dataset: Dataset to use for evaluation ('dev' or 'full')
        batch_size: Batch size for evaluation
        verbose: Whether to print detailed progress
    """
    # Initialize the model
    reformulator = QueryReformulator(model_size=model_size)
    device = reformulator.device
    
    # Load the test dataset
    dataset_path = Path(f"datasets/{dataset}.jsonl")
    test_data = load_dataset(dataset_path)
    
    if verbose:
        print(f"[INFO] Loaded {len(test_data)} examples from {dataset_path}")
    
    # Prepare for evaluation
    all_predictions = []
    all_references = []
    total_time = 0
    
    # Process examples
    for example in tqdm(test_data, desc="Evaluating"):
        query = example.get("query", "")
        reference = example.get("subqueries", "")
        
        # Measure inference time
        start_time = time.time()
        predictions = reformulator.reformulate(query)
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        
        # Store results
        all_predictions.append(predictions[0])  # Take first prediction
        all_references.append(reference)
        
        if verbose and len(all_predictions) % 10 == 0:
            print(f"Example {len(all_predictions)}:")
            print(f"  Query: {query}")
            print(f"  Prediction: {predictions[0]}")
            print(f"  Reference: {reference}")
            print(f"  Inference time: {inference_time:.4f}s")
    
    # Calculate BERTScore
    import numpy as np
    from bert_score import score
    
    P, R, F1 = score(all_predictions, all_references, lang="en", 
                     model_type="microsoft/roberta-large", device=device)
    
    # Prepare results
    results = {
        "bertscore_f1": F1.mean().item(),
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "avg_inference_time": total_time / len(test_data),
        "total_examples": len(test_data),
    }
    
    print(f"[INFO] Evaluation results: {results}")
    
    # Save results to file
    output_dir = Path(f"./benchmark/results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / f"{model_size}_{dataset}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results saved to {output_file}")
    
    return results

def benchmark_model(
    model_type: str,
    model_size: str,
    dataset: List[Dict[str, Any]],
    force_cpu: bool = False,
    use_onnx: bool = False,
    use_torchscript: bool = False,
):
    """Benchmark a model's inference speed and accuracy."""
    # This is a placeholder for your existing benchmark_model function
    # You can implement this based on your requirements
    pass

@click.group()
def main():
    """Benchmark and evaluate query reformulation models."""
    pass

@main.command()
@click.option('--model-size', type=click.Choice(['small', 'base', 'large']), default='small',
              help='Size of the T5 model to use')
@click.option('--dataset', type=str, default='dev',
              help='Dataset to use for evaluation (dev or full)')
@click.option('--verbose', is_flag=True, help='Print detailed progress')
def eval(model_size, dataset, verbose):
    """Evaluate a trained query reformulation model using the specified parameters."""
    print(f"[INFO] Evaluating with model_size={model_size}, dataset={dataset}")
    evaluate(model_size, dataset, verbose=verbose)

@main.command()
@click.option('--model-type', type=click.Choice(MODEL_TYPES), default='t5',
              help='Type of model to benchmark')
@click.option('--model-size', type=click.Choice(['small', 'base', 'large']), default='small',
              help='Size of the model to benchmark')
@click.option('--force-cpu', is_flag=True, help='Force CPU usage even if GPU is available')
@click.option('--use-onnx', is_flag=True, help='Use ONNX runtime for inference')
@click.option('--use-torchscript', is_flag=True, help='Use TorchScript for inference')
def benchmark(model_type, model_size, force_cpu, use_onnx, use_torchscript):
    """Benchmark a model's inference speed and accuracy."""
    dataset = load_dataset(DEV_DATASET)
    benchmark_model(
        model_type=model_type,
        model_size=model_size,
        dataset=dataset,
        force_cpu=force_cpu,
        use_onnx=use_onnx,
        use_torchscript=use_torchscript,
    )

if __name__ == "__main__":
    main()
