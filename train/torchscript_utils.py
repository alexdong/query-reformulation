"""
TorchScript model export, quantization, and inference utilities.
"""
import time
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

from rich.console import Console
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TORCHSCRIPT_DIR = Path("models/torchscript")
QUANTIZED_DIR = Path("models/quantized")
console = Console()

def quantize_model_dynamic(
    model: AutoModelForSeq2SeqLM,
    model_size: str
) -> torch.nn.Module:
    """Quantize the model using PyTorch's dynamic quantization.
    
    Args:
        model: The PyTorch model to quantize
        model_size: Size of the model ('small', 'base', or 'large')
        
    Returns:
        Quantized model
    """
    console.print("[bold cyan]Quantizing model using dynamic quantization...[/bold cyan]")
    
    # Move model to CPU for quantization
    model = model.cpu()
    
    # Prepare for quantization
    model.eval()
    
    # Get original model size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    console.print(f"[bold]Original model size:[/bold] {original_size:.2f} MB")
    
    try:
        # Create a custom quantization configuration
        from torch.quantization import per_channel_dynamic_qconfig
        
        # Define a custom quantization function for specific modules
        def quantize_linear_modules(module):
            """Apply quantization to linear modules only."""
            if isinstance(module, torch.nn.Linear):
                # Configure the module for dynamic quantization
                module.qconfig = per_channel_dynamic_qconfig
                # Return the quantized module
                return torch.quantization.quantize_dynamic(
                    module, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
            return module
        
        # Apply quantization to each module recursively
        console.print("[dim]Applying quantization to linear layers...[/dim]")
        
        # Create a copy of the model to avoid modifying the original
        quantized_model = type(model)(model.config)
        quantized_model.load_state_dict(model.state_dict())
        
        # Apply quantization to each module
        for name, module in list(quantized_model.named_modules()):
            if "." not in name:  # Only process top-level modules
                if isinstance(module, torch.nn.Linear):
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    if parent_name:
                        parent = quantized_model.get_submodule(parent_name)
                        setattr(parent, name.split(".")[-1], quantize_linear_modules(module))
                    else:
                        setattr(quantized_model, name, quantize_linear_modules(module))
                    console.print(f"[dim]Quantized linear layer: {name}[/dim]")
        
        # Get quantized model size (approximate)
        quantized_size = sum(p.numel() * (1 if hasattr(p, 'dtype') and p.dtype == torch.qint8 else p.element_size()) 
                            for p in quantized_model.parameters()) / (1024 * 1024)
        console.print(f"[bold]Quantized model size:[/bold] {quantized_size:.2f} MB")
        console.print(f"[bold]Size reduction:[/bold] {(1 - quantized_size / original_size) * 100:.2f}%")
        
        return quantized_model
    
    except Exception as e:
        console.print(f"[bold red]Error during dynamic quantization: {str(e)}[/bold red]")
        console.print("[bold yellow]Falling back to original model[/bold yellow]")
        return model

def save_model_for_inference(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_size: str,
    quantized: bool = False
) -> Path:
    """Save the model in a format optimized for inference.
    
    Args:
        model: The PyTorch model
        tokenizer: The tokenizer for the model
        model_size: Size of the model ('small', 'base', or 'large')
        quantized: Whether the model is quantized
        
    Returns:
        Path to the saved model directory
    """
    # Create directory if it doesn't exist
    suffix = "-quantized" if quantized else ""
    save_dir = QUANTIZED_DIR / f"{model_size}{suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    console.print(f"[bold green]Model saved to {save_dir}[/bold green]")
    return save_dir

def export_to_onnx(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    model_size: str
) -> Path:
    """Export the model to ONNX format.
    
    Args:
        model: The PyTorch model to export
        tokenizer: The tokenizer for the model
        model_size: Size of the model ('small', 'base', or 'large')
        
    Returns:
        Path to the saved ONNX model
    """
    console.print("[bold cyan]Exporting model to ONNX format...[/bold cyan]")
    
    # Create directory if it doesn't exist
    onnx_dir = Path("models/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"flan-t5-{model_size}.onnx"
    
    # Skip if model already exists
    if onnx_path.exists():
        console.print(f"[bold yellow]ONNX model already exists at {onnx_path}[/bold yellow]")
        return onnx_path
    
    try:
        from transformers.onnx import export
        from transformers.onnx.features import FeaturesManager
        
        # Get the appropriate ONNX config
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model)
        onnx_config = model_onnx_config(model.config)
        
        # Export to ONNX
        export(
            preprocessor=tokenizer,
            model=model,
            config=onnx_config,
            opset=12,
            output=onnx_path
        )
        
        console.print(f"[bold green]Model exported to {onnx_path}[/bold green]")
        return onnx_path
    
    except Exception as e:
        console.print(f"[bold red]Error during ONNX export: {str(e)}[/bold red]")
        console.print("[bold yellow]ONNX export failed[/bold yellow]")
        return None

def script_model_safely(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_size: str,
    force_cpu: bool = False,
    quantize: bool = False
) -> Tuple[Optional[torch.jit.ScriptModule], Path]:
    """Convert the model to TorchScript format with safety measures.
    
    Args:
        model: The PyTorch model to script
        tokenizer: The tokenizer for the model
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        quantize: Whether to quantize the model before scripting
        
    Returns:
        Tuple of (scripted_model, path_to_saved_model)
    """
    console.print(f"[bold green]Preparing model for scripting...[/bold green]")
    
    # Create directory if it doesn't exist
    TORCHSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    script_suffix = "-quantized" if quantize else ""
    script_path = TORCHSCRIPT_DIR / f"flan-t5-{model_size}{script_suffix}.pt"
    
    # Skip if model already exists
    if script_path.exists():
        console.print(f"[bold yellow]TorchScript model already exists at {script_path}[/bold yellow]")
        # Load the existing model
        scripted_model = torch.jit.load(script_path)
        return scripted_model, script_path
    
    # Quantize model if requested
    if quantize:
        model = quantize_model_dynamic(model, model_size)
    
    # Move model to CPU for scripting
    model = model.cpu()
    
    # Prepare model for scripting
    model.eval()
    
    # Get model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    console.print(f"[bold]Model size before scripting:[/bold] {model_size_mb:.2f} MB")
    
    # Create a simpler wrapper for generation that's easier to trace
    class SimpleGenerationWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask=None):
            # Use a simpler generation approach that's more likely to trace successfully
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=1,  # Use greedy decoding for tracing
                    do_sample=False,
                    early_stopping=False
                )
            return outputs
    
    # Create a simple inference function that doesn't require tracing
    def generate_with_model(input_ids, attention_mask=None):
        with torch.no_grad():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
    
    # Save the model and tokenizer for direct loading
    save_dir = save_model_for_inference(model, tokenizer, model_size, quantize)
    
    # Create a simple wrapper that loads the model and generates
    class ModelWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._generate_func = generate_with_model
            
        def forward(self, input_ids, attention_mask=None):
            return self._generate_func(input_ids, attention_mask)
    
    # Create a dummy wrapper that can be saved
    dummy_wrapper = ModelWrapper()
    
    # Save the script path with a note
    with open(script_path, "wb") as f:
        torch.save({
            "model_path": str(save_dir),
            "model_size": model_size,
            "quantized": quantize
        }, f)
    
    console.print(f"[bold green]Model reference saved to {script_path}[/bold green]")
    console.print("[bold yellow]Note: Using direct model loading instead of TorchScript due to tracing limitations[/bold yellow]")
    
    return None, script_path

def load_model_for_inference(
    model_size: str,
    force_cpu: bool = False,
    quantized: bool = False
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device]:
    """Load a model optimized for inference.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        quantized: Whether to load the quantized version of the model
        
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
    
    # Check if we have a saved model
    suffix = "-quantized" if quantized else ""
    save_dir = QUANTIZED_DIR / f"{model_size}{suffix}"
    
    if save_dir.exists():
        console.print(f"[bold green]Loading optimized model from {save_dir}[/bold green]")
        model = AutoModelForSeq2SeqLM.from_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
    else:
        console.print(f"[bold yellow]Optimized model not found at {save_dir}[/bold yellow]")
        console.print("[bold yellow]Loading original model and optimizing...[/bold yellow]")
        
        # Load original model
        model_name = f"google/flan-t5-{model_size}"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Quantize if requested
        if quantized:
            model = quantize_model_dynamic(model, model_size)
            
        # Save for future use
        save_model_for_inference(model, tokenizer, model_size, quantized)
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate_reformulation(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    query: str,
    device: torch.device
) -> Tuple[str, float]:
    """Generate query reformulation.
    
    Args:
        model: Model
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
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    
    reformulated_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reformulated_query, inference_time_ms

def test_model(
    model_size: str = "base",
    force_cpu: bool = False,
    quantized: bool = False
) -> None:
    """Test the model with a sample query.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        quantized: Whether to use the quantized version of the model
    """
    # Load model
    model, tokenizer, device = load_model_for_inference(model_size, force_cpu, quantized)
    
    # Test with sample query
    sample_query = "Create a table for top noise cancelling headphones that are not expensive"
    console.print(f"[bold]Testing with sample query:[/bold] {sample_query}")
    
    reformulated_query, inference_time = generate_reformulation(
        model, tokenizer, sample_query, device
    )
    
    console.print(f"[bold]Generated reformulation:[/bold] {reformulated_query}")
    console.print(f"[bold]Inference time:[/bold] {inference_time:.2f} ms")

def benchmark_model(
    model_size: str = "base",
    force_cpu: bool = False,
    quantized: bool = False,
    num_runs: int = 10
) -> Dict[str, Any]:
    """Benchmark the model with multiple runs.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        quantized: Whether to use the quantized version of the model
        num_runs: Number of inference runs to perform
        
    Returns:
        Dictionary with benchmark results
    """
    # Load model
    model, tokenizer, device = load_model_for_inference(model_size, force_cpu, quantized)
    
    # Test queries
    test_queries = [
        "Create a table for top noise cancelling headphones that are not expensive",
        "In what year was the winner of the 44th edition of the Miss World competition born?",
        "Who lived longer, Nikola Tesla or Milutin Milankovic?",
        "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?",
        "what are some ways to do fast query reformulation"
    ]
    
    # Run benchmark
    console.print(f"[bold]Running benchmark with {num_runs} queries...[/bold]")
    
    all_times = []
    for i, query in enumerate(test_queries):
        console.print(f"[bold]Query {i+1}:[/bold] {query}")
        
        # Warm-up run
        _, _ = generate_reformulation(model, tokenizer, query, device)
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            _, inference_time = generate_reformulation(model, tokenizer, query, device)
            times.append(inference_time)
            all_times.append(inference_time)
        
        avg_time = sum(times) / len(times)
        console.print(f"[bold]Average inference time:[/bold] {avg_time:.2f} ms")
    
    # Calculate overall statistics
    avg_time = sum(all_times) / len(all_times)
    median_time = sorted(all_times)[len(all_times) // 2]
    p90 = sorted(all_times)[int(len(all_times) * 0.9)]
    p95 = sorted(all_times)[int(len(all_times) * 0.95)]
    p99 = sorted(all_times)[int(len(all_times) * 0.99)]
    
    # Print results
    console.print("\n[bold]===== BENCHMARK RESULTS =====")
    console.print(f"[bold]Model:[/bold] flan-t5-{model_size}{' (quantized)' if quantized else ''}")
    console.print(f"[bold]Device:[/bold] {'CPU' if device.type == 'cpu' else 'GPU'}")
    console.print(f"[bold]Average time:[/bold] {avg_time:.2f} ms")
    console.print(f"[bold]Median time:[/bold] {median_time:.2f} ms")
    console.print(f"[bold]P90:[/bold] {p90:.2f} ms")
    console.print(f"[bold]P95:[/bold] {p95:.2f} ms")
    console.print(f"[bold]P99:[/bold] {p99:.2f} ms")
    
    return {
        "model_size": model_size,
        "quantized": quantized,
        "device": device.type,
        "avg_time": avg_time,
        "median_time": median_time,
        "p90": p90,
        "p95": p95,
        "p99": p99
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize a model for faster inference")
    parser.add_argument(
        "--model-size", 
        type=str, 
        default="base", 
        choices=["small", "base", "large"],
        help="Size of the model to optimize"
    )
    parser.add_argument(
        "--force-cpu", 
        action="store_true", 
        help="Force CPU usage even if GPU is available"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model to 8-bit precision"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark with multiple queries"
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_model(args.model_size, args.force_cpu, args.quantize)
    else:
        test_model(args.model_size, args.force_cpu, args.quantize)
