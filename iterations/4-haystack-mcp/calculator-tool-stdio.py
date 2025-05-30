"""Calculator Tool Example."""
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Calculator Tool")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Add an subtraction tool
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


if __name__ == "__main__":
    mcp.run(transport="stdio")
