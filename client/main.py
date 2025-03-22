import gradio as gr
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define model paths
hugging_face_model_name = 'alexdong/query-reformulation-knowledge-graph'
local_model_path = 'models/sft-small'
model_name = local_model_path

# Try to load from local path first, fall back to Hugging Face
print(f"Loading model from local path: {model_name}")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the function that will be run for every input
def generate_text(input_text):
    # output time used as well, ai!
    input_ids = tokenizer(f"reformulate: {input_text}", return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=120)
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(decoded_output)
    return decoded_output

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_text,         
    inputs="text",            
    outputs="text",           
    title="Query Reformulation",
    description="Enter a search query to see how the model rewrites it into RAG friendly subqueries.", # Description
)

# Display the interface
iface.launch(share=True)
