import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

hugging_face_model_name = 'alexdong/query-reformulation-knowledge-graph'
local_model_path = 'models/sft-small'
model_name = local_model_path

# Global variables for model and tokenizer
tokenizer = None
model = None

def load_model(model_size: str = "small") -> None:
    """Load the model and tokenizer only once"""
    global tokenizer, model, model_name
    
    # Update model path based on selected size
    local_model_path = f'models/sft-{model_size}'
    
    if tokenizer is None or model is None or local_model_path != model_name:
        print(f"Loading model from local path: {local_model_path}")
        tokenizer = T5Tokenizer.from_pretrained(local_model_path)
        
        # Load model with optimizations for inference
        model = T5ForConditionalGeneration.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16,  # Use half precision
            device_map="auto",           # Automatically choose best device
        )
        # Set to evaluation mode
        model.eval()
        
        # Update the current model name
        model_name = local_model_path

# Load model on import
load_model()

# Define the function that will be run for every input
def generate_text(input_text: str) -> str:
    """Generate reformulated queries from input text"""
    # Ensure model is loaded
    if tokenizer is None or model is None:
        load_model()
        
    with torch.no_grad():  # Disable gradient calculation for inference
        input_ids = tokenizer(f"reformulate: {input_text}", return_tensors="pt").input_ids
        
        # Move to same device as model
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
        if torch.mps.is_available():
            input_ids = input_ids.to('mps')
            
        # Generate with optimized parameters
        output_ids = model.generate(
            input_ids,
            max_length=120,
            num_beams=4,
            early_stopping=True,
        )
        
        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded_output
