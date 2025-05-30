"""Weather Client for MCP."""

import asyncio
from contextlib import AsyncExitStack
import logging

import dotenv
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


dotenv.load_dotenv()

llm = OpenAIChatGenerator(
    model="gpt-4o-mini",
)

messages = [ChatMessage.from_system("Be a simple chat bot.")]


def convert_mcp_tool_to_haystack_tool(mcp_tool, f):
    """Convert MCP tool to Haystack tool."""

    async def _f(x):
        """Call the tool function."""
        logging.info("Calling tool: %s with args: %s", mcp_tool.name, x)
        return f((mcp_tool.name, x))

    return Tool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        parameters=mcp_tool.inputSchema,
        function=lambda x: asyncio.run(_f(x)),
    )


class MCPClient:
    """MCP Client for connecting to a server and using tools."""

    def __init__(self):
        """Initialize session and client objects."""
        self.session: ClientSession | None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script (.py or .js)

        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logging.info(
            "\nConnected to server with tools: %s", [tool.name for tool in tools]
        )

        # List available resources
        response = await self.session.list_resources()
        resources = response.resources
        logging.info(
            "\nConnected to server with resources: %s",
            [resource.name for resource in resources],
        )

        # List available prompts
        response = await self.session.list_prompts()
        prompts = response.prompts
        logging.info(
            "\nConnected to server with prompts: %s",
            [prompt.name for prompt in prompts],
        )

    async def generate(self) -> str:
        """Generate response using available tools."""

        tools = [
            convert_mcp_tool_to_haystack_tool(
                tool, lambda args: self.session.call_tool(args.tool, args.x)
            )
            for tool in (await self.session.list_tools()).tools
        ]
        logging.info("Available tools: %s", [tool.name for tool in tools])

        response = llm.run(messages=messages, tools=tools)
        for reply in response["replies"]:
            messages.append(reply)
            if reply.meta["finish_reason"] == "tool_calls":
                response = await self.session.call_tool(
                    reply.tool_call.tool_name, reply.tool_call.arguments
                )
                messages.append(
                    ChatMessage.from_tool(
                        tool_result=response.content[0].text, origin=reply.tool_call
                    )
                )
                await self.generate()

    async def chat_loop(self):
        """Run an interactive chat loop."""
        logging.info("\nMCP Client Started!")
        logging.info("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                messages.append(ChatMessage.from_user(query))
                await self.generate()
                for message in messages:
                    print(  # noqa: T201
                        f"{message.role}: {message.text or ""}{message.tool_call or ""}{message.tool_call_result or ""}"
                    )

            except Exception:
                LOGGER.exception("\nError occurred while processing the query.")

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


async def main():
    """Run main function to run the client."""
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")  # noqa: T201
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
