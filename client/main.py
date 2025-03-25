import gradio as gr

from client.generate import generate_text, load_model


def generate_with_model(model_size: str, input_text: str) -> str:
    """Generate reformulated queries using the selected model size"""
    # Load the selected model if needed
    load_model(model_size)
    # Generate text using the loaded model
    return generate_text(input_text)

# Define the Gradio interface with model size dropdown
iface = gr.Interface(
    fn=generate_with_model,
    inputs=[
        gr.Dropdown(
            choices=["small", "base", "large"],
            label="Model Size",
            value="small",
        ),
        gr.Textbox(
            lines=3,
            placeholder="Enter your search query here...",
            label="Query",
        ),
    ],
    outputs=gr.Textbox(label="Reformulated Queries"),
    title="Query Reformulation",
    description="Enter a search query to see how it can be reformulated into more effective search engine queries.",
    examples=[
        ["base", "In what year was the winner of the 44th edition of the Miss World competition born?"],
        ["base", "Who lived longer, Nikola Tesla or Milutin Milankovic?"],
        ["base", "Create a table for top noise cancelling headphones that are not expensive"],
        ["base", "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"],
        ["base", "what are some ways to do fast query reformulation"],
    ],
)

if __name__ == "__main__":
    iface.launch()
