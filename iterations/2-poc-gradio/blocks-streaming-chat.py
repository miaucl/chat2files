"""Block streaming chat."""

import random
import time

import gradio as gr

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def user(user_message, history: list):
        """User function to handle user input and update chat history."""
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history: list):
        """Bot function to simulate bot response with streaming effect."""
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        history.append({"role": "assistant", "content": ""})
        for character in bot_message:
            history[-1]["content"] += character
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.launch()
