MODEL_CLASSES = ['small', 'base', 'large', '3b']
PEFT_CLASSES = ['3b']

def get_model_path(model_size: str) -> str:
    assert model_size in MODEL_CLASSES, f"Invalid model size: {model_size}"
    return f"./models/{'peft' if model_size in PEFT_CLASSES else 'sft'}-{model_size}"

def get_quantized_model_path(model_size: str) -> str:
    assert model_size in MODEL_CLASSES, f"Invalid model size: {model_size}"
    return get_model_path(model_size) + "-8bit"
