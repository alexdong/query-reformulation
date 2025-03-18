
from train.capabilities import get_backend_device

device = get_backend_device()

model_size = "base" if device == "cuda" else "small"
training_dataset = "datasets/full.jsonl" if device == "cuda" else "datasets/dev.jsonl"
model_name = f"google/flan-5t-{model_size}"
