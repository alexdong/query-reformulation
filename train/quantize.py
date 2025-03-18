"""
Quantize a model using PyTorch's FX Graph Mode Quantization.

This script provides a command-line interface to quantize a model
and benchmark its performance.

Example usage:
    python train/quantize.py --model-size base --benchmark
"""

import argparse
from train.torchscript_utils import (
    load_torchscript_model,
    benchmark_torchscript_model,
    test_torchscript_model
)

def main():
    parser = argparse.ArgumentParser(description="Quantize a model for faster inference")
    parser.add_argument(
        "--model-size", 
        type=str, 
        default="base", 
        choices=["small", "base", "large"],
        help="Size of the model to quantize"
    )
    parser.add_argument(
        "--force-cpu", 
        action="store_true", 
        help="Force CPU usage even if GPU is available"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark with multiple queries"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare quantized model with original model"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Run benchmark on original model
        print("\n===== ORIGINAL MODEL =====")
        original_results = benchmark_torchscript_model(
            args.model_size, 
            args.force_cpu, 
            quantized=False
        )
        
        # Run benchmark on quantized model
        print("\n===== QUANTIZED MODEL =====")
        quantized_results = benchmark_torchscript_model(
            args.model_size, 
            args.force_cpu, 
            quantized=True
        )
        
        # Print comparison
        print("\n===== PERFORMANCE COMPARISON =====")
        speedup = original_results["avg_time"] / quantized_results["avg_time"]
        print(f"Speed improvement: {speedup:.2f}x faster")
        
    elif args.benchmark:
        benchmark_torchscript_model(args.model_size, args.force_cpu, quantized=True)
    else:
        test_torchscript_model(args.model_size, args.force_cpu, quantized=True)

if __name__ == "__main__":
    main()
