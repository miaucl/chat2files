"""Tool."""

import argparse
import importlib.util
import logging
from pathlib import Path
import sys

from mcp.server.fastmcp import FastMCP

# run this server first before running the client mcp_filtered_tools.py or mcp_client.py
# it shows how easy it is to create a MCP server in just a few lines of code
# then we'll use the MCPTool to invoke the server

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DIR = Path(__file__).resolve().parent

# Load module from file
spec = importlib.util.spec_from_file_location(
    "retrieving_pipeline", DIR / "retrieving.py"
)
retrieving_pipeline = importlib.util.module_from_spec(spec)
sys.modules["retrieving_pipeline"] = retrieving_pipeline
spec.loader.exec_module(retrieving_pipeline)

mcp = FastMCP("MCP Tool")


@mcp.tool(
    name="ask_files",
    description="Ask a question and retrieve answers from indexed files.",
)
def ask_files(question: str) -> str:
    """Return a response from the retrieved files."""
    LOGGER.info("Running pipeline with question: %s", question)
    response = retrieving_pipeline.run(
        {
            "embedder": {"text": question},
            "retriever": {"top_k": 5},
            "reader": {"query": question, "top_k": 3},
        }
    )
    LOGGER.info(
        "Found %d documents with following answers",
        len(response["retriever"]["documents"]),
    )
    LOGGER.info("=" * 80)
    for answer in response["reader"]["answers"]:
        LOGGER.info("Answer: %s", answer.data)
        LOGGER.info("Document: %s", answer.document)
        LOGGER.info("Score: %s", answer.score)
        LOGGER.info("-" * 80)

    return (
        response["reader"]["answers"][0].data
        if response["reader"]["answers"]
        else "No answer found."
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run an MCP server with different transport options (sse or streamable-http)"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="streamable-http",
        choices=["sse", "streamable-http"],
        help="Transport mechanism for the MCP server (default: streamable-http)",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)
