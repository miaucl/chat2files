"""Dummy Tool Example."""

import argparse

from mcp.server.fastmcp import FastMCP

# run this server first before running the client mcp_filtered_tools.py or mcp_client.py
# it shows how easy it is to create a MCP server in just a few lines of code
# then we'll use the MCPTool to invoke the server


mcp = FastMCP("MCP Dummy Tool")


@mcp.tool()
def dummy(a: int, b: int) -> int:
    """Return a dummy message."""
    return "Im am a dummy tool, I do nothing but return this message."


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run an MCP server with different transport options (sse or streamable-http)"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "streamable-http"],
        help="Transport mechanism for the MCP server (default: sse)",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)
