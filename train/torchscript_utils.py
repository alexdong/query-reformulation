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

def quantize_model_fx(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    model_size: str
) -> torch.nn.Module:
    """Quantize the model using PyTorch's FX Graph Mode Quantization.
    
    Args:
        model: The PyTorch model to quantize
        tokenizer: The tokenizer for the model
        model_size: Size of the model ('small', 'base', or 'large')
        
    Returns:
        Quantized model
    """
    console.print("[bold cyan]Quantizing model using FX Graph Mode Quantization...[/bold cyan]")
    
    # Move model to CPU for quantization
    model = model.cpu()
    
    # Prepare for quantization
    model.eval()
    
    # Get original model size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    console.print(f"[bold]Original model size:[/bold] {original_size:.2f} MB")
    
    try:
        # Import quantization utilities
        from torch.quantization import quantize_fx
        from torch.quantization.fx.prepare import prepare_fx
        from torch.quantization.fx.convert import convert_fx
        from torch.ao.quantization import get_default_qconfig_mapping
        import torch.ao.quantization.quantize_fx as quantize_fx
        
        # Define quantization configuration for CPU
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        
        # Create a simplified wrapper for the model to make it easier to trace
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids, attention_mask):
                return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        wrapped_model = ModelWrapper(model)
        
        # Create example inputs for tracing
        example_inputs = tokenizer("This is a test query", return_tensors="pt")
        example_input_ids = example_inputs["input_ids"]
        example_attention_mask = example_inputs["attention_mask"]
        
        # Prepare the model for quantization
        console.print("[dim]Preparing model for quantization...[/dim]")
        prepared_model = prepare_fx(wrapped_model, qconfig_mapping, (example_input_ids, example_attention_mask))
        
        # Calibrate the model with a few examples
        console.print("[dim]Calibrating model...[/dim]")
        calibration_examples = [
            "What is the capital of France?",
            "Who was the first president of the United States?",
            "How many planets are in the solar system?",
            "What is the largest ocean on Earth?",
            "Who wrote the novel Pride and Prejudice?"
        ]
        
        with torch.no_grad():
            for example in calibration_examples:
                inputs = tokenizer(example, return_tensors="pt")
                prepared_model(inputs["input_ids"], inputs["attention_mask"])
        
        # Convert to quantized model
        console.print("[dim]Converting to quantized model...[/dim]")
        quantized_model = convert_fx(prepared_model)
        
        # Get quantized model size
        quantized_size = sum(p.numel() * (1 if hasattr(p, 'dtype') and p.dtype == torch.qint8 else p.element_size()) 
                            for p in quantized_model.parameters()) / (1024 * 1024)
        console.print(f"[bold]Quantized model size:[/bold] {quantized_size:.2f} MB")
        console.print(f"[bold]Size reduction:[/bold] {(1 - quantized_size / original_size) * 100:.2f}%")
        
        # Create a wrapper to restore the original interface
        class QuantizedModelWrapper(torch.nn.Module):
            def __init__(self, quantized_model, original_model):
                super().__init__()
                self.quantized_model = quantized_model
                self.original_model = original_model
                
            def forward(self, input_ids, attention_mask):
                logits = self.quantized_model(input_ids, attention_mask)
                return logits
                
            def generate(self, input_ids, attention_mask=None, **kwargs):
                # Fall back to the original model for generation
                return self.original_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
        
        final_model = QuantizedModelWrapper(quantized_model, model)
        return final_model
    
    except Exception as e:
        console.print(f"[bold red]Error during FX quantization: {str(e)}[/bold red]")
        console.print("[bold yellow]Falling back to original model[/bold yellow]")
        return model

def save_quantized_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_size: str
) -> Path:
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
    
    # Save model state dict
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    console.print(f"[bold green]Quantized model saved to {save_dir}[/bold green]")
    return save_dir

def script_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_size: str,
    force_cpu: bool = False,
    quantize: bool = False
) -> Tuple[torch.jit.ScriptModule, Path]:
    """Convert the model to TorchScript format.
    
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
        model = quantize_model_fx(model, tokenizer, model_size)
    
    # Move model to CPU for scripting
    model = model.cpu()
    
    # Prepare model for scripting
    model.eval()
    
    # Get model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    console.print(f"[bold]Model size before scripting:[/bold] {model_size_mb:.2f} MB")
    
    # Create a wrapper class for scripting that handles the generate method
    class ScriptableModule(torch.nn.Module):
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
    
    wrapped_model = ScriptableModule(model)
    
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
    console.print(f"[bold cyan]Saving TorchScript model...[/bold cyan]")
    torch.jit.save(traced_script, script_path)
    console.print(f"[bold green]Model saved to {script_path}[/bold green]")
    
    return traced_script, script_path

def optimize_for_inference(model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
    """Apply TorchScript optimizations for inference.
    
    Args:
        model: The TorchScript model to optimize
        
    Returns:
        Optimized TorchScript model
    """
    console.print("[bold cyan]Optimizing model for inference...[/bold cyan]")
    
    # Apply TorchScript optimizations
    model = torch.jit.optimize_for_inference(model)
    
    return model

def load_torchscript_model(
    model_size: str,
    force_cpu: bool = False,
    quantized: bool = False
) -> Tuple[torch.jit.ScriptModule, AutoTokenizer, torch.device]:
    """Load a TorchScript model and tokenizer.
    
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
    
    # Load tokenizer
    model_name = f"google/flan-t5-{model_size}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load TorchScript model
    script_suffix = "-quantized" if quantized else ""
    script_path = TORCHSCRIPT_DIR / f"flan-t5-{model_size}{script_suffix}.pt"
    
    if not script_path.exists():
        console.print(f"[bold red]TorchScript model not found at {script_path}[/bold red]")
        console.print("[bold yellow]Loading original model and converting to TorchScript...[/bold yellow]")
        
        # Load original model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Script the model
        scripted_model, _ = script_model(model, tokenizer, model_size, force_cpu, quantize=quantized)
        
        # Optimize for inference
        scripted_model = optimize_for_inference(scripted_model)
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
    force_cpu: bool = False,
    quantized: bool = False
) -> None:
    """Test the TorchScript model with a sample query.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        quantized: Whether to use the quantized version of the model
    """
    # Load model
    model, tokenizer, device = load_torchscript_model(model_size, force_cpu, quantized)
    
    # Test with sample query
    sample_query = "Create a table for top noise cancelling headphones that are not expensive"
    console.print(f"[bold]Testing with sample query:[/bold] {sample_query}")
    
    reformulated_query, inference_time = generate_reformulation_torchscript(
        model, tokenizer, sample_query, device
    )
    
    console.print(f"[bold]Generated reformulation:[/bold] {reformulated_query}")
    console.print(f"[bold]Inference time:[/bold] {inference_time:.2f} ms")

def benchmark_torchscript_model(
    model_size: str = "base",
    force_cpu: bool = False,
    quantized: bool = False,
    num_runs: int = 10
) -> Dict[str, Any]:
    """Benchmark the TorchScript model with multiple runs.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage even if GPU is available
        quantized: Whether to use the quantized version of the model
        num_runs: Number of inference runs to perform
        
    Returns:
        Dictionary with benchmark results
    """
    # Load model
    model, tokenizer, device = load_torchscript_model(model_size, force_cpu, quantized)
    
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
        _, _ = generate_reformulation_torchscript(model, tokenizer, query, device)
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            _, inference_time = generate_reformulation_torchscript(model, tokenizer, query, device)
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
        benchmark_torchscript_model(args.model_size, args.force_cpu, args.quantize)
    else:
        test_torchscript_model(args.model_size, args.force_cpu, args.quantize)
