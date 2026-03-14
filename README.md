# 🤖 AI Engineering Portfolio

> Built while learning AI engineering from scratch — 7 projects across
> LLM APIs, Prompt Engineering, RAG, and AI Agents.
> **Stack:** Python · FastAPI · LangChain · ChromaDB · Streamlit · Groq API

---

## 🗂️ Projects

### 1. 💬 Smart CLI Chatbot — `chatbot.py`
Multi-turn conversational chatbot with persistent memory and swappable personas.
- Conversation history management (token-aware trimming)
- Dynamic persona switching via config
- **Stack:** Python, Groq API, python-dotenv

---

### 2. 🗄️ SQL Query Explainer — `sql_explainer.py`
Paste any SQL query and get a plain-English explanation, issues found,
and an optimized rewrite.
- Structured JSON output from LLM
- Issues and suggestions panel
- **Stack:** Python, Groq API, Streamlit

---

### 3. 📊 Data Analyst Agent — `data_analyst.py`
Upload any CSV → ask questions in plain English → get answers + charts.
- LLM generates and executes pandas code dynamically
- Plotly chart rendering
- Agent reasoning visible to user
- **Stack:** Python, Groq API, Pandas, Plotly, Streamlit

---

### 4. 🧪 Prompt Eval Dashboard — `prompt_eval.py`
Scientifically compare prompt versions against test cases with automated scoring.
- 4 prompt variants tested against 10 test cases each
- Accuracy, latency, and score leaderboard
- Per-case pass/fail breakdown
- **Stack:** Python, Groq API, Streamlit

---

### 5. 📚 Knowledge Base Q&A — `knowledge_base.py`
Upload PDFs or text files → ask questions → get answers with source citations.
- Local embeddings via sentence-transformers
- ChromaDB vector database (persistent)
- Similarity scores per retrieved chunk
- **Stack:** Python, ChromaDB, Sentence-Transformers, Groq API, Streamlit

---

### 6. 🏢 Company Policy Chatbot — `policy_chatbot.py`
Conversational RAG chatbot — full chat memory combined with document retrieval.
- Multi-turn conversation with follow-up question understanding
- Retrieval-Augmented Generation with source citations
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