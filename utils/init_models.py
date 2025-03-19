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
    model_name = f"google/flan-t5-{model_size}" if not use_sft_model else f"./models/sft-{model_size}"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return device, tokenizer, model
