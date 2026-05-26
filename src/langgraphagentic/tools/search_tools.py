# src/langgraphagentic/tools/search_tools.py

from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
import os


def get_tools(tavily_api_key: str = None) -> list:
    """
    Returns list of tools for the chatbot.
    """
    api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY not found. "
            "Set environment variable TAVILY_API_KEY or pass it to get_tools()."
        )

    tools = [
        TavilySearch(
            api_key=api_key,
            max_results=5,
            include_answer=True,
            include_raw_content=False,
            include_images=True,
            search_depth="advanced",
        )
    ]
    return tools


def create_tool_node(tools: list) -> ToolNode:
    """Creates and returns a ToolNode for the LangGraph graph."""
    return ToolNode(tools=tools)