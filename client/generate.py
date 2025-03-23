import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

hugging_face_model_name = 'alexdong/query-reformulation-knowledge-graph'
local_model_path = 'models/sft-small'
model_name = local_model_path

# Global variables for model and tokenizer
tokenizer = None
model = None

def load_model():
    """Load the model and tokenizer only once"""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        print(f"Loading model from local path: {model_name}")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Load model with optimizations for inference
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,  # Use half precision
            device_map="auto"           # Automatically choose best device
        )
        # Set to evaluation mode
        model.eval()

# Load model on import
load_model()

# Define the function that will be run for every input
def generate_text(input_text) -> str:
    """Generate reformulated queries from input text"""
    # Ensure model is loaded
    if tokenizer is None or model is None:
        load_model()
        
    with torch.no_grad():  # Disable gradient calculation for inference
        input_ids = tokenizer(f"reformulate: {input_text}", return_tensors="pt").input_ids
        
        # Move to same device as model
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
            
        # Generate with optimized parameters
        output_ids = model.generate(
            input_ids, 
            max_length=120,
            num_beams=4,
            early_stopping=True
        )
        
        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded_output
