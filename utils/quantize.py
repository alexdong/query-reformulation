import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from models import MODEL_CLASSES, PEFT_CLASSES

def quantize_pytorch_native(model_size: str) -> None:
    """
    Loads a T5 model, quantizes it using PyTorch's dynamic quantization,
    and saves both the model and tokenizer to the specified directory.

    Args:
        model_size (str): The size of the T5 model (e.g., "small", "base").
        output_dir (str): The directory to save the quantized model and tokenizer.
    """
    assert model_size in MODEL_CLASSES, f"Invalid model size: {model_size}"
    model_name = f"./models/{'peft' if model_size in PEFT_CLASSES else 'sft'-{model_size}"
    output_dir = f"{model_name}-8bit"

    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Move model to CPU for quantization
    model = model.cpu()

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize only linear layers
        dtype=torch.qint8
    )

    os.makedirs(output_dir, exist_ok=True)
    quantized_model.config.save_pretrained(output_dir)
    torch.save(quantized_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)

    print(f"PyTorch quantized model and tokenizer saved to: {output_dir}")

    # Optional: Print memory usage comparison
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    quantized_size = sum(p.numel() * (1 if p.dtype == torch.qint8 else p.element_size())
                         for p in quantized_model.parameters()) / (1024 * 1024)

    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")

if __name__ == "__main__":
    model_size = "small"  # Change to "base", "large", etc. as needed
    quantize(model_size, output_directory)
