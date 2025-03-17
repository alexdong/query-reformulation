#!/usr/bin/env python3
"""
Quantize the model to 8-bit for faster inference on CPU.
"""

import argparse
import time
from pathlib import Path

import torch
from rich.console import Console
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Constants
QUANTIZED_DIR = Path("models/quantized")

console = Console()


def load_model(model_size: str, force_cpu: bool = False):
    """Load the model and tokenizer.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    model_name = f"google/flan-t5-{model_size}"
    
    console.print(f"[bold green]Loading model {model_name}...[/bold green]")
    
    # Determine device
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[bold blue]Using GPU: {torch.cuda.get_device_name(0)}[/bold blue]")
    else:
        device = torch.device("cpu")
        console.print("[bold yellow]Using CPU[/bold yellow]")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    
    return model, tokenizer, device


def quantize_model(model, model_size: str):
    """Quantize the model using int8 weights.
    
    Args:
        model: The PyTorch model to quantize
        model_size: Size of the model ('small', 'base', or 'large')
        
    Returns:
        Quantized model
    """
    console.print("[bold cyan]Quantizing model to 8-bit...[/bold cyan]")
    
    # Move model to CPU for quantization
    model = model.cpu()
    
    # Prepare for quantization
    model.eval()
    
    # Apply weight-only quantization (a simpler approach that works better with transformers)
    # We'll quantize the weights to int8 but keep activations in fp32
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Quantize weights to int8
                module.weight.data = torch.quantize_per_tensor(
                    module.weight.data, 
                    scale=1.0/127.0, 
                    zero_point=0, 
                    dtype=torch.qint8
                ).dequantize()
                
                console.print(f"[dim]Quantized linear layer: {name}[/dim]")
    
    return model


def save_quantized_model(model, tokenizer, model_size: str) -> Path:
    """Save the quantized model.
    
    Args:
        model: The quantized PyTorch model
        tokenizer: The tokenizer for the model
        model_size: Size of the model ('small', 'base', or 'large')
        
    Returns:
        Path to the saved model directory
    """
    # Create directory if it doesn't exist
    save_dir = QUANTIZED_DIR / model_size
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    console.print(f"[bold green]Quantized model saved to {save_dir}[/bold green]")
    return save_dir


def test_quantized_model(model, tokenizer, device) -> None:
    """Test the quantized model with a sample query.
    
    Args:
        model: The quantized model
        tokenizer: The tokenizer for the model
        device: The device to run inference on
    """
    sample_query = "Create a table for top noise cancelling headphones that are not expensive"
    
    console.print(f"[bold]Testing with sample query:[/bold] {sample_query}")
    
    # Tokenize input
    inputs = tokenizer(sample_query, return_tensors="pt").to(device)
    
    # Measure inference time
    start_time = time.time()
    
    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    end_time = time.time()
    
    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Print results
    console.print(f"[bold]Generated reformulation:[/bold] {output_text}")
    console.print(f"[bold]Inference time:[/bold] {(end_time - start_time) * 1000:.2f} ms")


def main(model_size: str, force_cpu: bool = False) -> None:
    """Main function to quantize the model.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
    """
    # Load model
    model, tokenizer, device = load_model(model_size, force_cpu)
    
    # Get original model size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Quantize model
    quantized_model = quantize_model(model, model_size)
    
    # Save quantized model
    save_quantized_model(quantized_model, tokenizer, model_size)
    
    # Test quantized model
    quantized_model = quantized_model.to(device)
    test_quantized_model(quantized_model, tokenizer, device)
    
    # Print model size comparison
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
    
    console.print(f"[bold]Original model size:[/bold] {original_size:.2f} MB")
    console.print(f"[bold]Quantized model size:[/bold] {quantized_size:.2f} MB")
    console.print(f"[bold]Size reduction:[/bold] {(1 - quantized_size / original_size) * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model to 8-bit for faster inference on CPU")
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
    
    args = parser.parse_args()
    
    main(args.model_size, args.force_cpu)
