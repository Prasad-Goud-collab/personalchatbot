# src/langgraphagentic/graph/graph_builder.py

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from src.langgraphagentic.states.state import State
from src.langgraphagentic.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagentic.nodes.chatbot_with_tool_node import ChatbotWithToolNode, create_chatbot
from src.langgraphagentic.nodes.ai_news_summarizer_node import ResearchNode, SummarizerNode
from src.langgraphagentic.nodes.linkedin_bot_node import LinkedInRetrieverNode, LinkedInAnswerNode
from src.langgraphagentic.tools.search_tools import get_tools, create_tool_node


class GraphBuilder:
    def __init__(self, model):
        self.llm = model
        self.graph_builder = StateGraph(State)

    def basic_chatbot_build_graph(self):
        """Flow: START → chatbot → END"""
        obj_basic_chatbot_node = BasicChatbotNode(self.llm)
        self.graph_builder.add_node("chatbot", obj_basic_chatbot_node.process)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)

    def chatbot_with_tools_build_graph(self, tavily_api_key: str = None):
        """Flow: START → chatbot → (tool call?) → tools → chatbot → END"""
        tools = get_tools(tavily_api_key=tavily_api_key)
        chatbot_node = create_chatbot(self.llm, tools)
        self.graph_builder.add_node("chatbot", chatbot_node)

        tool_node = create_tool_node(tools)
        self.graph_builder.add_node("tools", tool_node)

        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools", "chatbot")

    def ai_news_summarizer_build_graph(self, tavily_api_key: str = None):
        """Flow: START → research → summarize → END"""
        research_node = ResearchNode(tavily_api_key=tavily_api_key)
        summarizer_node = SummarizerNode(self.llm)

        self.graph_builder.add_node("research", research_node.process)
        self.graph_builder.add_node("summarize", summarizer_node.process)

        self.graph_builder.add_edge(START, "research")
        self.graph_builder.add_edge("research", "summarize")
        self.graph_builder.add_edge("summarize", END)

    def linkedin_bot_build_graph(self, faiss_store):
        """Flow: START → retrieve → answer → END"""
        retriever_node = LinkedInRetrieverNode(faiss_store)
        answer_node = LinkedInAnswerNode(self.llm)

        self.graph_builder.add_node("retrieve", retriever_node.process)
        self.graph_builder.add_node("answer", answer_node.process)

        self.graph_builder.add_edge(START, "retrieve")
        self.graph_builder.add_edge("retrieve", "answer")
        self.graph_builder.add_edge("answer", END)

    def set_up_graph(
        self,
        usecase: str,
        tavily_api_key: str = None,
        faiss_store=None
    ):
        """
        Sets up and compiles the graph for the selected use case.

        Args:
            usecase (str): Selected use case from UI.
            tavily_api_key (str): Optional Tavily API key.
            faiss_store: Optional FAISSVectorStore instance.

        Returns:
            CompiledGraph: The compiled LangGraph graph.
        """
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()

        elif usecase == "Chatbot With Web Search":
            self.chatbot_with_tools_build_graph(tavily_api_key=tavily_api_key)

        elif usecase == "AI News Summarizer":
            self.ai_news_summarizer_build_graph(tavily_api_key=tavily_api_key)

        elif usecase == "LinkedIn Profile Bot":
            if not faiss_store:
                raise RuntimeError("FAISSVectorStore is required for LinkedIn Profile Bot.")
            self.linkedin_bot_build_graph(faiss_store=faiss_store)

        return self.graph_builder.compile()