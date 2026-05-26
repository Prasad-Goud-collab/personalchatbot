# src/langgraphagentic/nodes/linkedin_bot_node.py

# src/langgraphagentic/nodes/linkedin_bot_node.py

from langchain_core.messages import HumanMessage, AIMessage
from src.langgraphagentic.states.state import State


class LinkedInRetrieverNode:
    """Retrieves relevant context from FAISS vectorstore."""

    def __init__(self, faiss_store):
        self.faiss_store = faiss_store

    def process(self, state: State) -> dict:
        """Retrieves relevant chunks based on user question."""
        question = state.get("question", "")

        if not question:
            return {
                "context": "",
                "messages": [AIMessage(content="No question provided.")]
            }

        # ✅ Retrieve more chunks
        context = self.faiss_store.retrieve(question, k=6)

        return {
            "context": context,
            "messages": [AIMessage(content=f"Context retrieved for: {question}")]
        }


class LinkedInAnswerNode:
    """Generates answer from retrieved context using LLM."""

    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """Generates answer using RAG context and conversation history."""
        question = state.get("question", "")
        context = state.get("context", "")
        chat_history = state.get("chat_history", [])

        if not context:
            return {
                "answer": "I could not find relevant information in the profile.",
                "messages": [AIMessage(content="I could not find relevant information in the profile.")]
            }

        # Build conversation history string
        history_text = ""
        for msg in chat_history[-6:]:
            if isinstance(msg, HumanMessage):
                history_text += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"

        # ✅ Better prompt — not too restrictive
        prompt = f"""You are a personal AI assistant with deep knowledge about a person's professional profile.

You have been given extracted content from their LinkedIn PDF profile below.
Answer the user's question in a helpful, conversational, and specific way.

Rules:
- Answer ONLY from the profile context provided
- If the exact answer is not in the context, say what you DO know that is related
- Be specific — mention names, dates, companies, skills exactly as they appear
- If truly nothing is relevant, say "I don't have that specific information in the profile"
- Never make up information

Previous Conversation:
{history_text if history_text else "No previous conversation."}

Profile Context:
{context}

User Question: {question}

Answer:"""

        result = self.llm.invoke([HumanMessage(content=prompt)])
        answer = result.content if hasattr(result, "content") else str(result)

        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "chat_history": [
                HumanMessage(content=question),
                AIMessage(content=answer)
            ]
        }