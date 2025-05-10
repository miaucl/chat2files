"""Simple Gradio demo that takes a name and an intensity level as input."""

import gradio as gr


def greet(name, intensity):
    """Return a greeting message with the specified intensity."""
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
