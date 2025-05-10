"""Chatbot demo using Gradio."""

import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")


def generate_text(text_prompt, history):
    """Generate text based on the input prompt."""
    response = generator(text_prompt, max_length=30, num_return_sequences=5)
    return response[0]["generated_text"]


textbox = gr.Textbox()

demo = gr.ChatInterface(generate_text)

demo.launch()
