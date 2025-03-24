import os

from transformers import BitsAndBytesConfig, T5ForConditionalGeneration, T5Tokenizer


def quantize(model_size: str, output_dir: str) -> None:
    """
    Loads a T5 model, quantizes it to 8-bit, and saves both the model
    and tokenizer to the specified directory.

    Args:
        model_size (str): The size of the T5 model (e.g., "small", "base").
        output_dir (str): The directory to save the quantized model and tokenizer.
    """
    model_name = f"google/flan-t5-{model_size}"

    # Use a consistent device_map setting
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0, # consider tuning if you are having issues
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_has_fp16_weight=False,
    )

    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # Crucial for optimal device placement
    )

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized model and tokenizer saved to: {output_dir}")

if __name__ == "__main__":
    model_size = "small"  # Change to "base", "large", etc. as needed
    output_directory = f"./models/flan-t5-8bit-{model_size}" # consistent path
    quantize(model_size, output_directory)
