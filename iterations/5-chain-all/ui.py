"""UI."""

import importlib
import logging
from pathlib import Path
import queue
import sys
import threading

import dotenv
import gradio as gr
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
dotenv.load_dotenv()

DIR = Path(__file__).resolve().parent


llm = OpenAIChatGenerator(
    model="gpt-4o-mini",
)

# Load module from file
spec = importlib.util.spec_from_file_location("client_pipeline", DIR / "client.py")
client_pipeline = importlib.util.module_from_spec(spec)
sys.modules["client_pipeline"] = client_pipeline
spec.loader.exec_module(client_pipeline)

initial_message = ChatMessage.from_assistant("Hi! How can I help you today?")


# Haystack -> Gradio
def haystack_to_gradio(messages: list[ChatMessage]) -> list[dict]:
    """Convert Haystack ChatMessage to Gradio message format."""
    return [{"role": m.role, "content": m.text} for m in messages]


with gr.Blocks() as ui:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type a message here...")
    state = gr.State([initial_message])
    clear = gr.ClearButton([msg, chatbot, state])

    def user(user_message, state: list):
        """User function to handle user input and update chat history."""
        new_state = state + [ChatMessage.from_user(user_message)]
        LOGGER.info("User message: %s", user_message)
        return "", haystack_to_gradio(new_state), new_state

    def bot(history: list, state: list):
        """Stream chat messages."""
        history.append(
            gr.ChatMessage(
                role="assistant",
                content="__Looking up the files...__",
                metadata={"title": "Initializing..."},
            )
        )
        q = queue.Queue()
        response = ""
        run_pipeline = client_pipeline.get_pipeline()
        client_pipeline.set_stream_callback(lambda message: q.put(message))
        t = threading.Thread(
            # target=llm.run, args=(state, lambda message: q.put(message))
            target=run_pipeline.run,
            args=(
                {
                    "llm": {"messages": state},
                    "adapter": {"initial_msg": state[-1:]},
                },
            ),
        )
        t.start()
        try:
            while True:
                try:
                    chunk = q.get_nowait()
                    if chunk is not None:
                        LOGGER.info("Received chunk: %s", chunk.content)
                        response += chunk.content
                        history[-1] = gr.ChatMessage(
                            role="assistant",
                            content=response,
                            metadata={"title": "Streaming..."},
                        )
                        yield history, state
                except queue.Empty:
                    if not t.is_alive() and q.empty():
                        LOGGER.info("Thread finished.")
                        break
            LOGGER.info("Assistant response: %s", response)
            new_state = state + [ChatMessage.from_assistant(response)]
            yield haystack_to_gradio(new_state), new_state
        except Exception as _e:
            LOGGER.exception("Error during bot processing")
            history[-1] = gr.ChatMessage(
                role="assistant",
                content="__Error occurred while processing your request.__",
            )
            yield history, state

    msg.submit(
        user, inputs=[msg, state], outputs=[msg, chatbot, state], queue=False
    ).then(bot, inputs=[chatbot, state], outputs=[chatbot, state])

    clear.click(
        lambda: (haystack_to_gradio([initial_message]), [initial_message]),
        outputs=[chatbot, state],
    )

    ui.load(
        lambda: (haystack_to_gradio([initial_message]), [initial_message]),
        outputs=[chatbot, state],
    )

ui.launch()
