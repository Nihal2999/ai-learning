# 🤖 AI Engineering Portfolio

> Built while learning AI engineering from scratch — 9 projects across
> LLM APIs, Prompt Engineering, RAG, Agents, and full-stack AI systems.
> **Stack:** Python · ChromaDB · Sentence-Transformers · Streamlit · Groq · Gemini · Tavily

---

## 🚀 Live Demos

| Project | Demo | Description |
|---|---|---|
| 🏢 Company Policy Chatbot | [▶ Live](https://ai-learning-bot.streamlit.app) | Conversational RAG over uploaded docs |
| 📊 Data Analyst Agent | [▶ Live](https://ai-learning-data-analyst.streamlit.app) | Upload CSV → ask questions → get charts |
| 🗄️ SQL Query Explainer | [▶ Live](https://ai-learning-sql-explainer.streamlit.app) | Paste SQL → get issues + optimized rewrite |
| 🧪 Prompt Eval Dashboard | [▶ Live](https://ai-learning-prompt-evaluation.streamlit.app) | A/B test prompts with automated scoring |
| 📚 Knowledge Base Q&A | [▶ Live](https://ai-learning-knowledge-base.streamlit.app) | Upload PDFs → semantic search → cited answers |
| 🔬 Research Agent | [▶ Live](https://ai-learning-research-agent.streamlit.app) | Topic → agent searches web → structured report |
| 📧 Email Triage Agent | [▶ Live](https://ai-learning-email-triage.streamlit.app) | Classify emails → extract info → draft replies |
| 🤖 AI Assistant (Capstone) | [▶ Live](https://ai-learning-assistant-capstone.streamlit.app) | RAG + web search + memory in one assistant |

---

## 🗂️ Projects

### 1. 💬 Smart CLI Chatbot — `chatbot.py`
Multi-turn conversational chatbot with persistent memory and swappable personas.
- Conversation history management (token-aware trimming)
- Dynamic persona switching via config
- **Stack:** Python, Groq API, python-dotenv

---

### 2. 🗄️ SQL Query Explainer — `sql_explainer.py`
🌐 **[Live Demo](https://ai-learning-sql-explainer.streamlit.app)**

Paste any SQL query and get a plain-English explanation, issues found, and an optimized rewrite.
- Structured JSON output from LLM
- Catches SELECT *, missing indexes, N+1 risks
- **Stack:** Python, Groq API, Streamlit

---

### 3. 📊 Data Analyst Agent — `data_analyst.py`
🌐 **[Live Demo](https://ai-learning-data-analyst.streamlit.app)**

Upload any CSV → ask questions in plain English → get answers + charts.
- LLM generates and executes pandas code dynamically
- Plotly chart rendering with agent reasoning visible
- **Stack:** Python, Groq API, Pandas, Plotly, Streamlit

---

### 4. 🧪 Prompt Eval Dashboard — `prompt_eval.py`
🌐 **[Live Demo](https://ai-learning-prompt-evaluation.streamlit.app)**

Scientifically compare prompt versions against test cases with automated scoring.
- 4 prompt variants tested against 10 test cases each
- Key finding: CoT + constraints (v3) = 100% accuracy. Minimal prompt (v4) = 0% and 18× slower
- **Stack:** Python, Groq API, Streamlit

---

### 5. 📚 Knowledge Base Q&A — `knowledge_base.py`
🌐 **[Live Demo](https://ai-learning-knowledge-base.streamlit.app/)**

Upload PDFs or text files → ask questions → get answers with source citations.
- Local embeddings via sentence-transformers (all-MiniLM-L6-v2)
- ChromaDB vector database with similarity scores
- **Stack:** Python, ChromaDB, Sentence-Transformers, Groq API, Streamlit

---

### 6. 🏢 Company Policy Chatbot — `policy_chatbot.py`
🌐 **[Live Demo](https://ai-learning-bot.streamlit.app)**

Conversational RAG chatbot — full chat memory combined with document retrieval.
- Multi-turn conversation with follow-up question understanding
- Grounded answers — never hallucinates beyond provided docs
- **Stack:** Python, ChromaDB, Sentence-Transformers, Groq API, Streamlit

---

### 7. 🔬 AI Research Agent — `research_agent.py`
🌐 **[Live Demo](https://ai-learning-research-agent.streamlit.app)**

Give any research topic → autonomous agent searches the web across multiple queries, reads full pages, and synthesises a structured report with key findings, perspectives, and cited sources.
- ReAct agent loop — autonomously decides which queries to run and which pages to read
- Structured report with executive summary, key findings, and perspectives
- Full agent reasoning log visible to user
- **Stack:** Python, Groq API, Tavily Search, Streamlit

---

### 8. 📧 Email Triage Agent — `email_triage.py`
🌐 **[Live Demo](https://ai-learning-email-triage.streamlit.app)**

Paste any email → AI classifies category and urgency, extracts key asks and deadlines, drafts a professional reply, and flags emails that need escalation.
- 8 categories: URGENT, BILLING, SUPPORT, SALES, HR, LEGAL, SPAM, INFO
- Extracts deadlines, sentiment, and sender intent automatically
- Triage dashboard showing processed email history and category breakdown
- **Stack:** Python, Groq API, Streamlit

---

### 9. 🤖 AI Assistant — Capstone — `assistant.py`
🌐 **[Live Demo](https://ai-learning-assistant-capstone.streamlit.app)**

Unified AI assistant combining every skill from the portfolio into one product. Autonomously routes between RAG document search, live web search, and direct LLM answers based on query context.
- Upload any PDF/TXT → ask questions grounded in your documents
- Searches the web for current information when needed
- Full conversation memory across all interactions
- Cites every source — document name or live URL
- Multi-provider LLM fallback — Groq → Gemini → OpenRouter → HuggingFace — automatic failover on rate limits
- Response caching — repeated queries served instantly with zero API calls
- **Stack:** Python, Groq, Gemini, OpenRouter, HuggingFace, ChromaDB, Sentence-Transformers, Tavily, Streamlit

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| LLM APIs | Groq (Llama 3.3 70B) |
| Vector DB | ChromaDB |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| UI | Streamlit |
| Data | Pandas, Plotly |
| Infra | Python venv, python-dotenv |
| Web Search | Tavily API |
| Multi-provider AI | Groq · Gemini · OpenRouter · HuggingFace |

---

## ⚙️ Setup

```bash
git clone https://github.com/Nihal2999/ai-learning.git
cd ai-learning
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your-key-here
```

Run any project:
```bash
streamlit run policy_chatbot.py
```

---

## 👤 About

**Nihal Vernekar** — Python Backend Developer, Bengaluru.
Transitioning into AI/ML Engineering.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/nihal-vernekar)
[![GitHub](https://img.shields.io/badge/GitHub-Nihal2999-black)](https://github.com/Nihal2999)