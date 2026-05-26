# src/langgraphagentic/nodes/ai_news_summarizer_node.py

from langchain_core.messages import HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from src.langgraphagentic.states.state import State
import os


class ResearchNode:
    """Searches the web for latest AI news on the given topic."""

    def __init__(self, tavily_api_key: str = None):
        api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY not found. "
                "Set environment variable TAVILY_API_KEY or pass it to ResearchNode()."
            )
        self.search_tool = TavilySearch(
            api_key=api_key,
            max_results=5,
            include_answer=True,
            include_raw_content=False,
            include_images=True,
            search_depth="advanced",
        )

    def process(self, state: State) -> dict:
        """Searches for latest AI news based on topic."""
        topic = state.get("topic", "latest AI news")
        query = f"latest AI news about {topic} 2025"

        results = self.search_tool.invoke(query)

        research_text = ""

        if isinstance(results, list):
            for r in results:
                title = r.get("title", "No title")
                content = r.get("content", "")
                url = r.get("url", "")
                research_text += f"\n\nTitle: {title}\nContent: {content}\nSource: {url}"

        elif isinstance(results, dict):
            for r in results.get("results", []):
                title = r.get("title", "No title")
                content = r.get("content", "")
                url = r.get("url", "")
                research_text += f"\n\nTitle: {title}\nContent: {content}\nSource: {url}"

        return {
            "research": research_text,
            "messages": [AIMessage(content=f"Research completed for topic: {topic}")]
        }


class SummarizerNode:
    """Summarizes the research into a clean AI news summary."""

    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """Generates a structured AI news summary from research."""
        topic = state.get("topic", "AI")
        research = state.get("research", "")

        if not research:
            return {
                "summary": "No research data found.",
                "messages": [AIMessage(content="No research data found.")]
            }

        prompt = f"""You are an expert AI news summarizer.

Based on the following research about "{topic}", write a clear, structured, and engaging news summary.

Format the summary exactly as follows:

## 🤖 AI News Summary: {topic}

### 📌 Key Highlights
- List the main news points as bullet points

### 📰 Detailed Summary
Write 3-4 paragraphs summarizing the news in detail

### 🔍 Key Takeaways
- List 3-5 important takeaways

### 🌐 Sources
List the source URLs from the research

Research Data:
{research}

Write a professional and engaging summary now:
"""

        result = self.llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if hasattr(result, "content") else str(result)

        return {
            "summary": summary,
            "messages": [AIMessage(content=summary)]
        }