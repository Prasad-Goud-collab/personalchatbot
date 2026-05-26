## src/langgraphagentic/main.py

# src/langgraphagentic/main.py

# src/langgraphagentic/main.py

import streamlit as st
from .UI.streamlit.loadui import LoadStreamlitUI
from .UI.streamlit.display_result import DisplayResultStreamlit
import os
from .LLMS.groqllm import GroqLLM
from .graph.graph_builder import GraphBuilder
from .vectorstore.faiss_store import FAISSVectorStore


# ✅ Set your PDF path here
LINKEDIN_PDF_PATH = "src\\langgraphagentic\\data\\kovi_2 Resume 2025.pdf"


def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    """
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return

    user_message = st.chat_input("Enter your message:")

    if not user_message:
        st.info("Type a message and press Enter to start.")
        return

    try:
        obj_llm_config = GroqLLM(user_controls_input=user_input)
        model = obj_llm_config.get_llm_model()
        if not model:
            st.error("Error: LLM model could not be initialized.")
            return

        usecase = user_input.get("selected_usecase", "")

        tavily_key = user_input.get("TAVILY_API_KEY")
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        # ✅ LinkedIn Profile Bot — load from backend path
        faiss_store = None
        if usecase == "LinkedIn Profile Bot":

            # ✅ Check PDF exists
            if not os.path.exists(LINKEDIN_PDF_PATH):
                st.error(
                    f"PDF not found at: `{LINKEDIN_PDF_PATH}`. "
                    "Please place your PDF in the data/ folder."
                )
                return

            # ✅ Build FAISS only once per session
            if "faiss_store" not in st.session_state:
                with st.spinner("📄 Loading LinkedIn Profile..."):
                    faiss_store = FAISSVectorStore()
                    faiss_store.build_from_pdf_path(LINKEDIN_PDF_PATH)
                    st.session_state.faiss_store = faiss_store

                    # ✅ Debug — shows extracted PDF chunks
                    with st.expander("🔍 Debug: PDF Content Extracted"):
                        docs = faiss_store.vectorstore.similarity_search(
                            "experience skills education", k=10
                        )
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)
                            st.divider()

                    st.success("✅ LinkedIn Profile loaded successfully!")
            else:
                faiss_store = st.session_state.faiss_store

        graph_builder = GraphBuilder(model)
        graph = graph_builder.set_up_graph(
            usecase=usecase,
            tavily_api_key=tavily_key,
            faiss_store=faiss_store
        )

        DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()

    except Exception as e:
        st.error(f"Error: {e}")
        return