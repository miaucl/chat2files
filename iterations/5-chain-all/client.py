"""Client for the ask file tool using Haystack MCP integration."""

import logging
from pathlib import Path

import dotenv
from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.mcp import (
    MCPToolset,
    SSEServerInfo,
    StreamableHttpServerInfo,
)

DIR = Path(__file__).resolve().parent

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


dotenv.load_dotenv()

llm = OpenAIChatGenerator(
    model="gpt-4o-mini",
)

# Create server info for the time service (can also use SSEServerInfo for remote servers)
server_info_sse = SSEServerInfo(
    url="http://localhost:8000/sse"
)  # Fallback to SSE if StreamableHttpServerInfo is not available
server_info_streamable = StreamableHttpServerInfo(url="http://localhost:8000/mcp")

# Create the toolset - this will automatically discover all available tools
# You can optionally specify which tools to include
mcp_toolset = MCPToolset(
    # server_info=server_info_sse,
    server_info=server_info_streamable,
)

stream_callback = lambda x: LOGGER.info("Streamed response: %s", x)

# Create a pipeline with the toolset
client_pipeline = Pipeline()
client_pipeline.add_component(
    "llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=mcp_toolset)
)
client_pipeline.add_component("tool_invoker", ToolInvoker(tools=mcp_toolset))
client_pipeline.add_component(
    "adapter",
    OutputAdapter(
        template="{{ initial_msg + initial_tool_messages + tool_messages }}",
        output_type=list[ChatMessage],
        unsafe=True,
    ),
)
client_pipeline.add_component(
    "response_llm",
    OpenAIChatGenerator(
        model="gpt-4o-mini", streaming_callback=lambda x: stream_callback(x)
    ),
)
client_pipeline.connect("llm.replies", "tool_invoker.messages")
client_pipeline.connect("llm.replies", "adapter.initial_tool_messages")
client_pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
client_pipeline.connect("adapter.output", "response_llm.messages")

client_pipeline.draw(str(DIR / "client.png"))


def get_pipeline() -> Pipeline:
    """Get the pipeline."""
    return client_pipeline


def set_stream_callback(cb):
    """Set the stream callback."""
    global stream_callback  # noqa: PLW0603
    stream_callback = cb
