#!/usr/bin/env python3
"""
Quantize the model to 8-bit for faster inference on CPU.
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
from rich.console import Console
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Constants
QUANTIZED_DIR = Path("models/quantized")

console = Console()


def load_model(model_size: str, force_cpu: bool = False) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """Load the model and tokenizer.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        
    Returns:
        Tuple of (model, tokenizer)
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
    
    return model, tokenizer


def quantize_model(model: AutoModelForSeq2SeqLM, model_size: str) -> torch.nn.Module:
    """Dynamically quantize the model to 8-bit.
    
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
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize only linear layers
        dtype=torch.qint8
    )
    
    return quantized_model


def save_quantized_model(model: torch.nn.Module, tokenizer: AutoTokenizer, model_size: str) -> Path:
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


def test_quantized_model(model: torch.nn.Module, tokenizer: AutoTokenizer) -> None:
    """Test the quantized model with a sample query.
    
    Args:
        model: The quantized model
        tokenizer: The tokenizer for the model
    """
    sample_query = "Create a table for top noise cancelling headphones that are not expensive"
    
    console.print(f"[bold]Testing with sample query:[/bold] {sample_query}")
    
    # Tokenize input
    inputs = tokenizer(sample_query, return_tensors="pt")
    
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
    model, tokenizer = load_model(model_size, force_cpu)
    
    # Quantize model
    quantized_model = quantize_model(model, model_size)
    
    # Save quantized model
    save_quantized_model(quantized_model, tokenizer, model_size)
    
    # Test quantized model
    test_quantized_model(quantized_model, tokenizer)
    
    # Print model size comparison
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
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
