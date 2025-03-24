import os
import sys
from typing import Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

if sys.platform == "linux":
    pass

from utils.quantize import quantize


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

    quantized_model_path = f"./models/flan-t5-8bit-{model_size}"
    
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    print(f"[DEBUG] Tokenizer loaded successfully, vocab size: {tokenizer.vocab_size}")

    if quantized and not os.path.exists(quantized_model_path) and sys.platform == "linux":
        quantize(model_size, quantized_model_path)

    if quantized and sys.platform == "linux":
        model = T5ForConditionalGeneration.from_pretrained(quantized_model_path, device_map="auto")
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    print(f"[DEBUG] Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
        
    return device, tokenizer, model
