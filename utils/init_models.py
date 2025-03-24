from typing import Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def get_backend_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def init_models(model_size: str, use_sft_model: bool = False) -> Tuple[str, T5Tokenizer, T5ForConditionalGeneration]:
    device = get_backend_device()
    print(f"[DEBUG] Using device: {device}")
    
    model_name = f"google/flan-t5-{model_size}" if not use_sft_model else f"./models/sft-{model_size}"
    print(f"[DEBUG] Loading model from: {model_name}")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        print(f"[DEBUG] Tokenizer loaded successfully, vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        raise
        
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print(f"[DEBUG] Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Check if model has any NaN parameters
        has_nan = False
        for name, param in model.named_parameters():
            if param.isnan().any():
                has_nan = True
                print(f"[WARNING] NaN values found in model parameter: {name}")
        assert not has_nan, "Model contains NaN parameters"
        
        # Move model to device
        model = model.to(device)
        print(f"[DEBUG] Model moved to {device}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise
        
    return device, tokenizer, model
