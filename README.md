# 🤖 AI Engineering Portfolio

> Built while learning AI engineering from scratch — 6 projects across
> LLM APIs, Prompt Engineering, RAG, and AI Agents.
> **Stack:** Python · ChromaDB · Sentence-Transformers · Streamlit · Groq API

---

## 🚀 Live Demos

| Project | Demo | Description |
|---|---|---|
| 🏢 Company Policy Chatbot | [▶ Live](https://ai-learning-bot.streamlit.app) | Conversational RAG over uploaded docs |
| 📊 Data Analyst Agent | [▶ Live](https://ai-learning-data-analyst.streamlit.app) | Upload CSV → ask questions → get charts |
| 🗄️ SQL Query Explainer | [▶ Live](https://ai-learning-sql-explainer.streamlit.app) | Paste SQL → get issues + optimized rewrite |
| 🧪 Prompt Eval Dashboard | [▶ Live](https://ai-learning-prompt-evaluation.streamlit.app) | A/B test prompts with automated scoring |
| 📚 Knowledge Base Q&A | [▶ Live](https://YOUR-KB-URL.streamlit.app) | Upload PDFs → semantic search → cited answers |

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

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| LLM APIs | Groq (Llama 3.3 70B) |
| Vector DB | ChromaDB |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| UI | Streamlit |
| Data | Pandas, Plotly |
| Infra | Python venv, python-dotenv |

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