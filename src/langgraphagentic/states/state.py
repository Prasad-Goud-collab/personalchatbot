# src/langgraphagentic/states/state.py

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    topic: Optional[str]
    research: Optional[str]
    summary: Optional[str]
    question: Optional[str]          # ✅ user question for linkedin bot
    context: Optional[str]           # ✅ retrieved RAG context
    answer: Optional[str]            # ✅ final answer from LLM
    chat_history: Annotated[list[BaseMessage], operator.add]  # ✅ conversation memory