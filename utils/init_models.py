import json
import torch
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
from typing import Tuple

def get_backend_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def init_models(use_sft_model: bool = False) -> Tuple[str, str, T5Tokenizer, T5ForConditionalGeneration]:
    device = get_backend_device()

    model_size = "base" if device == "cuda" else "small"

    model_name = f"google/flan-t5-{model_size}" if not use_sft_model else f"./models/t5-{model_size}-sft"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model_size, device, tokenizer, model
