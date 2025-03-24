import gradio as gr

from client.generate import generate_text

# Define the Gradio interface. 
# Add a drop down to select the model size: small, base, ai!
iface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="Query Reformulation",
    description="Enter a search query to see how the model rewrites it into RAG friendly subqueries.", # Description
)

# Display the interface
iface.launch(share=True)
