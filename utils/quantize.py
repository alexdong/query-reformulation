import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from models import MODEL_CLASSES, PEFT_CLASSES

def quantize(model_size: str) -> None:
    assert model_size in MODEL_CLASSES, f"Invalid model size: {model_size}"
    model_name = f"./models/{'peft' if model_size in PEFT_CLASSES else 'sft'-{model_size}"
    output_dir = f"{model_name}-8bit"
    print(f"[INFO] Quantizing model: {model_name} ...")

    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
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

    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    quantized_size = sum(p.numel() * (1 if p.dtype == torch.qint8 else p.element_size())
                         for p in quantized_model.parameters()) / (1024 * 1024)

    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")

if __name__ == "__main__":
    # Loop through all MODEL_CLASSES and quantize the models. If the model doesn't exist or output directory already exists, the script will skip, ai!
    model_size = "small"  # Change to "base", "large", etc. as needed
    quantize(model_size)
