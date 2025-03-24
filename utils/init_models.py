import sys
from typing import Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

if sys.platform == "linux":
    from transformers import BitsAndBytesConfig


def get_backend_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def init_models(model_size: str, quantized: bool = False) -> Tuple[str, T5Tokenizer, T5ForConditionalGeneration]:
    device = get_backend_device()
    
    model_name = f"google/flan-t5-{model_size}"
    print(f"[DEBUG] Loading model from: {model_name}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    print(f"[DEBUG] Tokenizer loaded successfully, vocab size: {tokenizer.vocab_size}")
        
    if quantized:
        model = T5ForConditionalGeneration.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    print(f"[DEBUG] Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
        
    return device, tokenizer, model
