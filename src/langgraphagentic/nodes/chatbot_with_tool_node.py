# src/langgraphagentic/nodes/chatbot_with_tool_node.py

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.langgraphagentic.states.state import State


class ChatbotWithToolNode:
    """Chatbot logic enhanced with tool integration."""

    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """Processes the input state and generates a response with tool integration."""
        messages = state.get("messages", [])
        normalized = self._normalize_messages(messages)

        if not normalized:
            return {"messages": [AIMessage(content="No input provided.")]}

        result = self.llm.invoke(normalized)
        return {"messages": [result] if not isinstance(result, list) else result}

    def _normalize_messages(self, messages: list) -> list:
        """Normalizes messages to proper LangChain message objects."""
        normalized = []
        for m in messages:
            if isinstance(m, BaseMessage):
                normalized.append(m)
            elif isinstance(m, str):
                normalized.append(HumanMessage(content=m))
            elif isinstance(m, tuple) and len(m) == 2:
                role, content = m
                if role == "human":
                    normalized.append(HumanMessage(content=content))
                else:
                    normalized.append(AIMessage(content=content))
        return normalized


def create_chatbot(llm, tools):
    """
    Returns a chatbot node function with tools bound to the LLM.
    """
    # Verify tools have names before binding
    for tool in tools:
        if not hasattr(tool, "name") or not tool.name:
            raise ValueError(
                f"Tool {tool} is missing a name. Groq requires all tools to have a name."
            )

    if hasattr(llm, "bind_tools"):
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    def chatbot_node(state: State) -> dict:
        """Chatbot node that processes state and returns LLM response."""
        messages = state.get("messages", [])

        normalized = []
        for m in messages:
            if isinstance(m, BaseMessage):
                normalized.append(m)
            elif isinstance(m, str):
                normalized.append(HumanMessage(content=m))
            elif isinstance(m, tuple) and len(m) == 2:
                role, content = m
                if role == "human":
                    normalized.append(HumanMessage(content=content))
                else:
                    normalized.append(AIMessage(content=content))

        if not normalized:
            return {"messages": [AIMessage(content="No input provided.")]}

        # Keep tool call messages even if content is empty
        normalized = [
            m for m in normalized
            if (hasattr(m, "content") and m.content not in (None, ""))
            or (hasattr(m, "tool_calls") and m.tool_calls)
        ]

        if not normalized:
            return {"messages": [AIMessage(content="No input provided.")]}

        result = llm_with_tools.invoke(normalized)
        return {"messages": [result] if not isinstance(result, list) else result}

    return chatbot_node