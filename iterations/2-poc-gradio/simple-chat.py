"""Simple chat."""

import dotenv
import gradio as gr
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

dotenv.load_dotenv()

messages = [ChatMessage.from_system("Be a simple chat bot.")]

llm = OpenAIChatGenerator(
    model="gpt-4o-mini",
)


def chatbot(message, history):
    """Stream chat messages."""
    messages.append(ChatMessage.from_user(message))
    response = llm.run(messages=messages)
    messages.append(response["replies"][0])
    return response["replies"][0].text


gr.ChatInterface(
    fn=chatbot,
    type="messages",
    examples=[
        "Can you tell me where Giorgio lives?",
        "What's the weather like in Madrid?",
        "Who lives in London?",
        "What's the weather like where Mark lives?",
    ],
    title="I am a simple chatbot!",
).launch()
