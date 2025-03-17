"""
TorchScript model export and inference utilities.
"""
import torch
from pathlib import Path
from typing import Union

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TORCHSCRIPT_DIR = Path("models/torchscript")

def save_as_torchscript(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    model_size: str
) -> Path:
    """Save the model as TorchScript.
    
    Args:
        model: The PyTorch model to save
        tokenizer: The tokenizer for the model
        model_size: Size of the model ('small', 'base', or 'large')
        
    Returns:
        Path to the saved TorchScript model
    """
    print(f"[INFO] Saving model as TorchScript...")
    
    # Create directory if it doesn't exist
    TORCHSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    script_path = TORCHSCRIPT_DIR / f"flan-t5-{model_size}.pt"
    
    # Skip if model already exists
    if script_path.exists():
        print(f"[INFO] TorchScript model already exists at {script_path}")
        return script_path
    
    # Prepare model for scripting
    model.eval()
    
    # Create a wrapper class for tracing that handles the generate method
    class TracedModule(torch.nn.Module):
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
    
    traced_model = TracedModule(model)
    
    # Create dummy input for tracing
    dummy_input = tokenizer("reformulate:This is a test query", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    
    # Trace the model
    with torch.no_grad():
        traced_script = torch.jit.trace(
            traced_model,
            (input_ids, attention_mask)
        )
    
    # Save the traced model
    torch.jit.save(traced_script, script_path)
    print(f"[INFO] Model saved to {script_path}")
    
    return script_path

def generate_reformulation_torchscript(
    model: torch.jit.ScriptModule,
    tokenizer: AutoTokenizer,
    query: str,
    device: torch.device
) -> str:
    """Generate query reformulation using a TorchScript model.
    
    Args:
        model: TorchScript model
        tokenizer: The tokenizer for the model
        query: The query to reformulate
        device: The device to run inference on
        
    Returns:
        The reformulated query
    """
    input_text = f"reformulate:{query}"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
