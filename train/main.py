import torch

from train.capabilities import get_capabilities
capabilities = get_capabilities()
model_size = "base" if capabilities["cuda_available"] else "small"
model_name = f"google/flan-5t-{model_size}"
