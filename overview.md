# LangGraph Agentic AI Chatbot

A production-grade Agentic AI Chatbot built using **LangGraph**, **LangChain**, and **Groq LLM** with multiple intelligent use cases including web search, AI news summarization, and a personal LinkedIn profile bot.

---

##  Features

| Use Case | Description |
|---|---|
| 💬 Basic Chatbot | Conversational AI powered by Groq LLM |
| 🔍 Chatbot With Web Search | Real-time web search using Tavily API |
| 📰 AI News Summarizer | Searches and summarizes latest AI news |
| 👤 LinkedIn Profile Bot | Chat with any LinkedIn profile using RAG |

---

##  Architecture

```
User Input (Streamlit UI)
        ↓
LangGraph State Machine
        ↓
┌─────────────────────────────────┐
│  Basic Chatbot  │  Web Search   │
│  News Summary   │  LinkedIn Bot │
└─────────────────────────────────┘
        ↓
Groq LLM (llama-3.3-70b-versatile)
        ↓
Response displayed on UI
```

---

##  Tech Stack

```
LangGraph        — Agentic graph orchestration
LangChain        — LLM framework
Groq             — Ultra-fast LLM inference
Tavily           — Real-time web search
FAISS            — Vector similarity search
HuggingFace      — Sentence embeddings
FastAPI          — Backend API
Streamlit        — Frontend UI
Python           — Core language
```

---

## Project Structure

```
src/langgraphagentic/
├── states/
│   └── state.py                   — LangGraph state schema
├── nodes/
│   ├── basic_chatbot_node.py      — Basic chat logic
│   ├── chatbot_with_tool_node.py  — Web search logic
│   ├── ai_news_summarizer_node.py — News summarizer
│   └── linkedin_bot_node.py       — RAG chatbot logic
├── tools/
│   └── search_tools.py            — Tavily search tool
├── vectorstore/
│   └── faiss_store.py             — PDF → FAISS pipeline
├── graph/
│   └── graph_builder.py           — Graph orchestration
├── LLMS/
│   └── groqllm.py                 — Groq LLM setup
└── UI/
    └── streamlit/
        ├── loadui.py              — Sidebar UI
        └── display_result.py      — Result rendering
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/langgraph-agentic-chatbot.git
cd langgraph-agentic-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API keys in sidebar
```
GROQ API Key   → https://console.groq.com
TAVILY API Key → https://app.tavily.com
```

### 4. Run the app
```bash
streamlit run app.py
```

---

##  API Keys Required

| Key | Purpose | Get it here |
|---|---|---|
| `GROQ_API_KEY` | LLM inference | console.groq.com |
| `TAVILY_API_KEY` | Web search | app.tavily.com |

---

##  How Each Use Case Works

### 💬 Basic Chatbot
```
User message → Groq LLM → Response
```

###  Chatbot With Web Search
```
User message → Tool call check → Tavily search → Groq LLM → Response
```

###  AI News Summarizer
```
Topic → Tavily searches latest news → Groq summarizes → Structured report
```

### LinkedIn Profile Bot
```
LinkedIn PDF → PyPDF extracts text → Chunks → FAISS embeddings
     ↓
User question → Semantic search → Relevant chunks → Groq answers
```

---

##  LangGraph Flow

```
START
  ↓
Select Usecase
  ↓
┌──────────────────────────────────────────────┐
│ Basic Chatbot   → chatbot → END              │
│ Web Search      → chatbot ⇄ tools → END      │
│ News Summarizer → research → summarize → END │
│ LinkedIn Bot    → retrieve → answer → END    │
└──────────────────────────────────────────────┘
```

---

##  Key Highlights

- ✅ Modular architecture — easy to add new use cases
- ✅ Production-grade code with proper error handling
- ✅ Real-time web search with source links and images
- ✅ RAG pipeline with semantic chunking and retrieval
- ✅ Conversation memory for LinkedIn bot
- ✅ Configurable via `config.ini` — no code changes needed
- ✅ Supports multiple Groq models via dropdown

---

## Models Supported

```
llama-3.3-70b-versatile          ← recommended
llama-3.1-8b-instant             ← fastest
meta-llama/llama-4-maverick      ← latest
openai/gpt-oss-120b              ← most powerful
```

---

## Requirements

```
langchain
langgraph
langchain-groq
langchain-tavily
langchain-huggingface
langchain-community
langchain-text-splitters
faiss-cpu
sentence-transformers
streamlit
pypdf
python-dotenv
```

---

## 🙋 Author

**Komara Prasad**  
AI Engineer | LLM Systems | RAG Pipelines | Agentic AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)(www.linkedin.com/in/prasadkpk)]
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Prasad-Goud-collab)
