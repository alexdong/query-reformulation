import gradio as gr

from client.generate import generate_text, load_model
from models import MODEL_CLASSES

DEFAULT_MODEL = 'large'

load_model(DEFAULT_MODEL)
with gr.Blocks(title="Query Reformulation") as app:
    model_size_dropdown = gr.Dropdown(
            choices=MODEL_CLASSES,
            label="Model Size",
            value=DEFAULT_MODEL,)

    model_size_dropdown.change(
            fn=lambda x: load_model(x),
            inputs=model_size_dropdown)

    query_input = gr.Textbox(
            lines=3,
            placeholder="Enter your search query here...",
            label="Query",)

    output = gr.Textbox(label="Reformulated Queries")

    generate_button = gr.Button("Generate")
    generate_button.click(
            fn=lambda x: generate_text(x),
            inputs=query_input,
            outputs=output)

    examples=gr.Examples(
            examples=[
                ["In what year was the winner of the 44th edition of the Miss World competition born?"],
                ["Who lived longer, Nikola Tesla or Milutin Milankovic?"],
                ["Create a table for top noise cancelling headphones that are not expensive"],
                ["Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"],
                ["what are some ways to do fast query reformulation"],
                ],
            inputs=query_input,
            )

    app.launch(share=True)
