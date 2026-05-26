# src/langgraphagentic/UI/streamlit/loadui.py
# src/langgraphagentic/UI/streamlit/loadui.py
# src/langgraphagentic/UI/streamlit/loadui.py

import streamlit as st
from .. import Config


class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        st.set_page_config(page_title=self.config.get_page_title(), layout="wide")
        st.header(self.config.get_page_title())

        with st.sidebar:
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            # LLM selection
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)

            if self.user_controls["selected_llm"] == "Groq":
                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox("Select Model", model_options)
                self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = st.text_input(
                    "GROQ API Key", type="password"
                )

                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning(
                        "Please enter your GROQ API key. "
                        "Don't have one? Refer: https://console.groq.com"
                    )

            # Usecase selection
            self.user_controls["selected_usecase"] = st.selectbox("Select Usecase", usecase_options)

            # Tavily API key
            if self.user_controls["selected_usecase"] in (
                "Chatbot With Web Search",
                "AI News Summarizer"
            ):
                self.user_controls["TAVILY_API_KEY"] = st.session_state["TAVILY_API_KEY"] = st.text_input(
                    "TAVILY API KEY", type="password"
                )
                if not self.user_controls["TAVILY_API_KEY"]:
                    st.warning(
                        "Please enter your TAVILY_API_KEY. "
                        "Don't have one? Refer: https://app.tavily.ai"
                    )

            # ✅ LinkedIn Bot — just show info, no upload
            if self.user_controls["selected_usecase"] == "LinkedIn Profile Bot":
                st.info("📄 LinkedIn Profile PDF loaded from backend.")

        return self.user_controls