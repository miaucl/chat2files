"""Dummy client for the calculator tool using Haystack MCP integration."""

import logging
from pathlib import Path

import dotenv
from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.mcp import MCPToolset, SSEServerInfo

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
server_info_sse = SSEServerInfo(url="http://localhost:8000/sse")

# Create the toolset - this will automatically discover all available tools
# You can optionally specify which tools to include
mcp_toolset = MCPToolset(
    server_info=server_info_sse,
)

# Create a pipeline with the toolset
pipeline = Pipeline()
pipeline.add_component(
    "llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=mcp_toolset)
)
pipeline.add_component("tool_invoker", ToolInvoker(tools=mcp_toolset))
pipeline.add_component(
    "adapter",
    OutputAdapter(
        template="{{ initial_msg + initial_tool_messages + tool_messages }}",
        output_type=list[ChatMessage],
        unsafe=True,
    ),
)
pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))
pipeline.connect("llm.replies", "tool_invoker.messages")
pipeline.connect("llm.replies", "adapter.initial_tool_messages")
pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
pipeline.connect("adapter.output", "response_llm.messages")

# Run the pipeline with a user question
user_input = "Get the dummy response from the dummy tool?"
user_input_msg = ChatMessage.from_user(text=user_input)

result = pipeline.run(
    {
        "llm": {"messages": [user_input_msg]},
        "adapter": {"initial_msg": [user_input_msg]},
    }
)
print(result["response_llm"]["replies"][0].text)  # noqa: T201

pipeline.draw(str(DIR / "dummy-client.png"))
