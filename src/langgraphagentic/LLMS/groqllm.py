import streamlit as st
import os
from langchain_groq import ChatGroq


class GroqLLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        try:
            groq_api_key = self.user_controls_input.get("GROQ_API_KEY", "").strip()
            selected_groq_model = self.user_controls_input.get("selected_groq_model", "").strip()

            if not groq_api_key and not os.environ.get("GROQ_API_KEY", ""):
                st.error("Please enter your GROQ API key.")
                return None

            if not selected_groq_model:
                st.error("Please select a GROQ model.")
                return None

            if not groq_api_key:
                groq_api_key = os.environ["GROQ_API_KEY"]

            try:
                llm = ChatGroq(api_key=groq_api_key, model=selected_groq_model)
                return llm
            except Exception as exc:
                message = str(exc)
                if "model_not_found" in message.lower() or "does not exist" in message.lower():
                    st.error(
                        f"Model '{selected_groq_model}' was not found or you do not have access to it. "
                        "Please choose a different model in the sidebar."
                    )
                    return None
                raise

        except Exception as e:
            raise ValueError(f"Error Occurred With Exception: {e}")
