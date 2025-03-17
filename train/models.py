"""
Common model loading and inference functions.
"""
import torch
from pathlib import Path
from typing import Union, Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_SIZES = ["small", "base", "large"]

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

def generate_reformulation(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    query: str,
    device: torch.device,
) -> str:
    """Generate query reformulation using the PyTorch model.
    
    Args:
        model: PyTorch model
        tokenizer: The tokenizer for the model
        query: The query to reformulate
        device: The device to run inference on
        
    Returns:
        The reformulated query
    """
    input_text = f"reformulate:{query}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_return_sequences=1,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
