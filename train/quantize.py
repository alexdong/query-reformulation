"""
Convert a model to TorchScript for faster inference.

This script provides a command-line interface to convert a model to TorchScript
and benchmark its performance.

Example usage:
    python train/quantize.py --model-size base --benchmark
"""

import argparse
from train.torchscript_utils import (
    test_torchscript_model,
    benchmark_torchscript_model
)

def main():
    parser = argparse.ArgumentParser(description="Convert a model to TorchScript for faster inference")
    parser.add_argument(
        "--model-size", 
        type=str, 
        default="base", 
        choices=["small", "base", "large"],
        help="Size of the model to convert"
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
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_torchscript_model(args.model_size, args.force_cpu)
    else:
        test_torchscript_model(args.model_size, args.force_cpu)

if __name__ == "__main__":
    main()
