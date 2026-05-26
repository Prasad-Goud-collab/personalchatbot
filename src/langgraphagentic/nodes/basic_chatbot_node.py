# src/langgraphagentic/nodes/basic_chatbot_node.py

# src/langgraphagentic/nodes/basic_chatbot_node.py

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.langgraphagentic.states.state import State


class BasicChatbotNode:
    """Basic Chatbot logic implementation."""

    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """Processes the input state and generates a chatbot response."""
        messages = state.get("messages", [])

        # Normalize to proper BaseMessage objects
        normalized = []
        for m in messages:
            if isinstance(m, BaseMessage):
                normalized.append(m)
            elif isinstance(m, str):
                normalized.append(HumanMessage(content=m))
            elif isinstance(m, tuple) and len(m) == 2:
                normalized.append(HumanMessage(content=str(m[1])))
            elif isinstance(m, dict):
                content = m.get("content") or m.get("text") or ""
                normalized.append(HumanMessage(content=str(content)))

        if not normalized:
            return {"messages": [AIMessage(content="No input provided.")]}

        result = self.llm.invoke(normalized)
        return {"messages": [result] if not isinstance(result, list) else result}