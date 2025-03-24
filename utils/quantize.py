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
    # Loop through all MODEL_CLASSES and quantize the models
    for model_size in MODEL_CLASSES:
        model_name = f"./models/{'peft' if model_size in PEFT_CLASSES else 'sft'}-{model_size}"
        output_dir = f"{model_name}-8bit"
        
        # Check if model exists
        if not os.path.exists(model_name):
            print(f"[WARNING] Model {model_name} does not exist, skipping...")
            continue
            
        # Check if output directory already exists
        if os.path.exists(output_dir):
            print(f"[WARNING] Output directory {output_dir} already exists, skipping...")
            continue
            
        print(f"[INFO] Processing model: {model_size}")
        try:
            quantize(model_size)
            print(f"[SUCCESS] Quantized model {model_size}")
        except Exception as e:
            print(f"[ERROR] Failed to quantize model {model_size}: {e}")
