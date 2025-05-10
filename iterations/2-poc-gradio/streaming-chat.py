"""Streaming chat."""

import queue
import threading

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
    q = queue.Queue()
    response = ""
    t = threading.Thread(
        target=llm.run, args=(messages, lambda message: q.put(message))
    )
    t.start()
    while True:
        chunk = q.get()
        if not t.is_alive():
            break
        if chunk is not None:
            response += chunk.content
            yield gr.ChatMessage(
                role="assistant", content=response, metadata={"title": "Streaming ..."}
            )
    messages.append(ChatMessage.from_assistant(response))
    yield gr.ChatMessage(role="assistant", content=messages[-1].text)


gr.ChatInterface(
    fn=chatbot,
    type="messages",
    examples=[
        "Count to 10.",
        "Can you tell me where Giorgio lives?",
        "What's the weather like in Madrid?",
        "Who lives in London?",
        "What's the weather like where Mark lives?",
    ],
    title="I am a simple *streaming* chatbot!",
).launch()
