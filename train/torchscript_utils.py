"""
TorchScript model export, quantization, and inference utilities.
"""
import time
import torch
from pathlib import Path
from typing import Tuple, Union, Optional

from rich.console import Console
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TORCHSCRIPT_DIR = Path("models/torchscript")
console = Console()

def quantize_and_script_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    model_size: str,
    force_cpu: bool = False
) -> Tuple[torch.jit.ScriptModule, Path]:
    """Quantize the model and save it as TorchScript.
    
    Args:
        model: The PyTorch model to quantize and save
        tokenizer: The tokenizer for the model
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        
    Returns:
        Tuple of (scripted_model, path_to_saved_model)
    """
    console.print(f"[bold green]Preparing model for quantization and scripting...[/bold green]")
    
    # Create directory if it doesn't exist
    TORCHSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    script_path = TORCHSCRIPT_DIR / f"flan-t5-{model_size}-quantized.pt"
    
    # Skip if model already exists
    if script_path.exists():
        console.print(f"[bold yellow]Quantized TorchScript model already exists at {script_path}[/bold yellow]")
        # Load the existing model
        scripted_model = torch.jit.load(script_path)
        return scripted_model, script_path
    
    # Move model to CPU for quantization
    model = model.cpu()
    
    # Prepare model for quantization
    model.eval()
    
    # Get original model size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    console.print(f"[bold]Original model size:[/bold] {original_size:.2f} MB")
    
    # Apply dynamic quantization
    console.print("[bold cyan]Applying dynamic quantization...[/bold cyan]")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize only linear layers
        dtype=torch.qint8
    )
    
    # Get quantized model size
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
    console.print(f"[bold]Quantized model size:[/bold] {quantized_size:.2f} MB")
    console.print(f"[bold]Size reduction:[/bold] {(1 - quantized_size / original_size) * 100:.2f}%")
    
    # Create a wrapper class for scripting that handles the generate method
    class QuantizedModule(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask):
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_return_sequences=1
            )
    
    wrapped_model = QuantizedModule(quantized_model)
    
    # Create dummy input for tracing
    console.print("[bold cyan]Tracing model with dummy input...[/bold cyan]")
    dummy_input = tokenizer("reformulate:This is a test query", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    
    # Trace the model
    with torch.no_grad():
        traced_script = torch.jit.trace(
            wrapped_model,
            (input_ids, attention_mask)
        )
    
    # Save the traced model
    console.print(f"[bold cyan]Saving quantized TorchScript model...[/bold cyan]")
    torch.jit.save(traced_script, script_path)
    console.print(f"[bold green]Model saved to {script_path}[/bold green]")
    
    return traced_script, script_path

def load_torchscript_model(
    model_size: str,
    force_cpu: bool = False
) -> Tuple[torch.jit.ScriptModule, AutoTokenizer, torch.device]:
    """Load a TorchScript model and tokenizer.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Determine device
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[bold blue]Using GPU: {torch.cuda.get_device_name(0)}[/bold blue]")
    else:
        device = torch.device("cpu")
        console.print("[bold yellow]Using CPU[/bold yellow]")
    
    # Load tokenizer
    model_name = f"google/flan-t5-{model_size}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load TorchScript model
    script_path = TORCHSCRIPT_DIR / f"flan-t5-{model_size}-quantized.pt"
    
    if not script_path.exists():
        console.print(f"[bold red]TorchScript model not found at {script_path}[/bold red]")
        console.print("[bold yellow]Loading original model and quantizing...[/bold yellow]")
        
        # Load original model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Quantize and script the model
        scripted_model, _ = quantize_and_script_model(model, tokenizer, model_size, force_cpu)
    else:
        console.print(f"[bold green]Loading TorchScript model from {script_path}[/bold green]")
        scripted_model = torch.jit.load(script_path, map_location=device)
    
    return scripted_model, tokenizer, device

def generate_reformulation_torchscript(
    model: torch.jit.ScriptModule,
    tokenizer: AutoTokenizer,
    query: str,
    device: torch.device
) -> Tuple[str, float]:
    """Generate query reformulation using a TorchScript model.
    
    Args:
        model: TorchScript model
        tokenizer: The tokenizer for the model
        query: The query to reformulate
        device: The device to run inference on
        
    Returns:
        Tuple of (reformulated_query, inference_time_ms)
    """
    input_text = f"reformulate:{query}"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    
    reformulated_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reformulated_query, inference_time_ms

def test_torchscript_model(
    model_size: str = "base",
    force_cpu: bool = False
) -> None:
    """Test the TorchScript model with a sample query.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
    """
    # Load model
    model, tokenizer, device = load_torchscript_model(model_size, force_cpu)
    
    # Test with sample query
    sample_query = "Create a table for top noise cancelling headphones that are not expensive"
    console.print(f"[bold]Testing with sample query:[/bold] {sample_query}")
    
    reformulated_query, inference_time = generate_reformulation_torchscript(
        model, tokenizer, sample_query, device
    )
    
    console.print(f"[bold]Generated reformulation:[/bold] {reformulated_query}")
    console.print(f"[bold]Inference time:[/bold] {inference_time:.2f} ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize and script a model for faster inference on CPU")
    parser.add_argument(
        "--model-size", 
        type=str, 
        default="base", 
        choices=["small", "base", "large"],
        help="Size of the model to quantize and script"
    )
    parser.add_argument(
        "--force-cpu", 
        action="store_true", 
        help="Force CPU usage even if GPU is available"
    )
    
    args = parser.parse_args()
    
    test_torchscript_model(args.model_size, args.force_cpu)
