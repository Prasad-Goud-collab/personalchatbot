# src/langgraphagentic/UI/streamlit/display_result.py

# src/langgraphagentic/UI/streamlit/display_result.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json


class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message

        if not user_message:
            return

        initial_state = {"messages": [HumanMessage(content=user_message)]}

        # ─── Basic Chatbot ────────────────────────────────────────────
        if usecase == "Basic Chatbot":
            for event in graph.stream(initial_state):
                if isinstance(event, dict):
                    for node_output in event.values():
                        messages = (
                            node_output.get("messages", [])
                            if isinstance(node_output, dict)
                            else []
                        )
                        for message in messages:
                            if isinstance(message, AIMessage) and message.content:
                                with st.chat_message("user"):
                                    st.write(user_message)
                                with st.chat_message("assistant"):
                                    st.write(message.content)

        # ─── Chatbot With Web Search ──────────────────────────────────
        elif usecase in ("Chatbot With Web Search", "Chatbot With Web", "Chatbot with Web"):
            res = graph.invoke(initial_state)

            for message in res.get("messages", []):

                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)

                elif isinstance(message, ToolMessage):
                    with st.chat_message("ai"):
                        st.markdown("🔍 **Web Search Results**")
                        try:
                            tool_data = json.loads(message.content)

                            if isinstance(tool_data, dict) and tool_data.get("answer"):
                                st.markdown(f"**Answer:** {tool_data['answer']}")

                            results = []
                            if isinstance(tool_data, list):
                                results = tool_data
                            elif isinstance(tool_data, dict):
                                results = tool_data.get("results", [])

                            for r in results:
                                title = r.get("title", "No title")
                                url = r.get("url", "")
                                content = r.get("content", "")
                                st.markdown(f"**🔗 [{title}]({url})**")
                                st.write(
                                    content[:300] + "..."
                                    if len(content) > 300
                                    else content
                                )
                                st.divider()

                            images = (
                                tool_data.get("images", [])
                                if isinstance(tool_data, dict)
                                else []
                            )
                            if images:
                                st.markdown("**🖼️ Images**")
                                cols = st.columns(min(3, len(images)))
                                for i, img in enumerate(images[:3]):
                                    img_url = (
                                        img if isinstance(img, str) else img.get("url", "")
                                    )
                                    if img_url:
                                        with cols[i]:
                                            st.image(img_url, use_container_width=True)

                        except (json.JSONDecodeError, TypeError):
                            st.write(message.content)

                elif isinstance(message, AIMessage) and message.content:
                    with st.chat_message("assistant"):
                        st.write(message.content)

        # ─── AI News Summarizer ───────────────────────────────────────
        elif usecase == "AI News Summarizer":
            initial_state = {
                "messages": [HumanMessage(content=user_message)],
                "topic": user_message,
                "research": None,
                "summary": None,
            }

            with st.spinner("🔍 Searching and summarizing latest AI news..."):
                res = graph.invoke(initial_state)

            summary = res.get("summary", "")
            research = res.get("research", "")

            with st.chat_message("user"):
                st.write(f"📰 Topic: {user_message}")

            with st.chat_message("assistant"):
                if summary:
                    st.markdown(summary)
                else:
                    st.warning("No summary generated. Try a different topic.")

            if research:
                with st.expander("📚 View Raw Research Sources"):
                    st.markdown(research)

        # ─── LinkedIn Profile Bot ─────────────────────────────────────
        elif usecase == "LinkedIn Profile Bot":

            # ✅ Initialize chat history in session state
            if "linkedin_chat_history" not in st.session_state:
                st.session_state.linkedin_chat_history = []

            # ✅ Display previous chat history
            for msg in st.session_state.linkedin_chat_history:
                if isinstance(msg, HumanMessage):
                    with st.chat_message("user"):
                        st.write(msg.content)
                elif isinstance(msg, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(msg.content)

            # ✅ Build state with question and history
            linkedin_state = {
                "messages": [HumanMessage(content=user_message)],
                "question": user_message,
                "context": None,
                "answer": None,
                "chat_history": st.session_state.linkedin_chat_history,
            }

            with st.spinner("🔍 Searching profile..."):
                res = graph.invoke(linkedin_state)

            answer = res.get("answer", "")
            context = res.get("context", "")

            # ✅ Show current exchange
            with st.chat_message("user"):
                st.write(user_message)

            with st.chat_message("assistant"):
                if answer:
                    st.markdown(answer)
                else:
                    st.warning("Could not find an answer in the profile.")

            # ✅ Update chat history
            st.session_state.linkedin_chat_history.extend([
                HumanMessage(content=user_message),
                AIMessage(content=answer)
            ])

            # ✅ Show retrieved context in expander for debugging
            if context:
                with st.expander("📄 View Retrieved Profile Context"):
                    st.markdown(context)