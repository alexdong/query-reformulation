import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model from Hugging Face
# update the below code so it loads from both local path and Hugging Face, ai!
model_name = 'alexdong/query-reformulation-knowledge-base-t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the function that will be run for every input
def generate_text(input_text):
    input_ids = tokenizer(f"reformulate: {input_text}", return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=50)
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
iface.launch()
