import os
import json
import time
import hashlib
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ─── Secret loader ────────────────────────────────────────────────
def get_secret(key: str):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

# ─── Tavily ───────────────────────────────────────────────────────
tavily_key    = get_secret("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

# ─── Multi-provider LLM setup ─────────────────────────────────────
def get_providers() -> list:
    providers = []

    groq_key = get_secret("GROQ_API_KEY")
    if groq_key:
        providers.append({
            "name":     "Groq",
            "client":   OpenAI(api_key=groq_key,
                               base_url="https://api.groq.com/openai/v1"),
            "model":    "llama-3.3-70b-versatile",
            "priority": 1
        })

    gemini_key = get_secret("GEMINI_API_KEY")
    if gemini_key:
        providers.append({
            "name":     "Gemini",
            "client":   OpenAI(api_key=gemini_key,
                               base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
            "model":    "gemini-3.1-flash-lite-preview",
            "priority": 2
        })
        
    openrouter_key = get_secret("OPENROUTER_API_KEY")
    if openrouter_key:
        providers.append({
            "name":     "OpenRouter",
            "client":   OpenAI(api_key=openrouter_key,
                               base_url="https://openrouter.ai/api/v1"),
            "model":    "openrouter/auto",
            "priority": 3
        })

    hf_key = get_secret("HF_API_KEY")
    if hf_key:
        providers.append({
            "name":     "HuggingFace",
            "client":   OpenAI(api_key=hf_key,
                               base_url="https://api-inference.huggingface.co/v1/"),
            "model":    "moonshotai/Kimi-K2-Instruct-0905",
            "priority": 4
        })

    return sorted(providers, key=lambda x: x["priority"])


def call_llm(messages, tools=None, temperature=0.3, max_tokens=1000):
    """
    Call LLM with automatic provider fallback.
    Priority: Groq → Gemini → OpenRouter → HuggingFace
    """
    providers  = get_providers()
    last_error = None

    if not providers:
        raise Exception("No API keys found. Add GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, or HF_API_KEY.")

    for provider in providers:
        try:
            kwargs = {
                "model":       provider["model"],
                "temperature": temperature,
                "max_tokens":  max_tokens,
                "messages":    messages
            }
            if tools:
                kwargs["tools"]       = tools
                kwargs["tool_choice"] = "auto"

            response = provider["client"].chat.completions.create(**kwargs)
            response._provider_name = provider["name"]
            return response

        except Exception as e:
            last_error = e
            err = str(e).lower()

            # Rate limit / quota → next provider immediately
            if any(x in err for x in ["429", "rate_limit", "quota", "resource_exhausted"]):
                continue

            # Server errors → short wait then next
            elif any(x in err for x in ["500", "502", "503", "timeout"]):
                time.sleep(1)
                continue

            # Tool calling failed → retry without tools on same provider
            elif tools and any(x in err for x in ["tool", "function", "400"]):
                try:
                    no_tool_kwargs = {k: v for k, v in kwargs.items()
                                      if k not in ("tools", "tool_choice")}
                    response = provider["client"].chat.completions.create(**no_tool_kwargs)
                    response._provider_name = provider["name"]
                    return response
                except:
                    continue

            # Other error → next provider
            else:
                continue

    raise Exception(f"All providers failed. Last error: {last_error}")


# ─── Config ───────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 3
MAX_HISTORY   = 6

# ─── Tools ────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search uploaded knowledge base documents for relevant information. Use when the user asks about something that might be in their uploaded files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information, news, or facts not in the documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query"}
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT = f"""You are a helpful AI assistant with access to two tools:
1. search_documents — searches the user's uploaded knowledge base
2. search_web — searches the web for current information

Today's date: {datetime.now().strftime('%B %d, %Y')}

Guidelines:
- Questions about uploaded docs → use search_documents
- Current events or topics not in docs → use search_web
- General knowledge → answer directly without tools
- Always cite sources (document name or URL)
- Be concise and helpful"""


# ─── Embedding + ChromaDB ─────────────────────────────────────────
@st.cache_resource
def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name="assistant_kb",
        embedding_function=ef
    )


# ─── Document ingestion ───────────────────────────────────────────
def ingest_file(file, collection) -> int:
    name = file.name
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        text   = "\n".join(p.extract_text() for p in reader.pages)
    else:
        text = file.read().decode("utf-8")

    if not text.strip():
        return 0

    chunks, start, idx = [], 0, 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append({
                "id":     f"{name}_c{idx}",
                "text":   chunk,
                "source": name
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    collection.upsert(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks]
    )
    return len(chunks)


# ─── Tool execution ───────────────────────────────────────────────
def execute_tool(name, args, collection):
    if name == "search_documents":
        if collection.count() == 0:
            return "No documents uploaded yet.", {}
        results = collection.query(
            query_texts=[args["query"]],
            n_results=min(TOP_K, collection.count())
        )
        chunks = [
            f"[{results['metadatas'][0][i]['source']} | "
            f"{round((1 - results['distances'][0][i]) * 100, 1)}% match]\n"
            f"{results['documents'][0][i]}"
            for i in range(len(results["documents"][0]))
        ]
        return "\n\n".join(chunks) or "No relevant documents found.", \
               {"type": "document", "query": args["query"]}

    elif name == "search_web":
        if not tavily_client:
            return "Web search unavailable — TAVILY_API_KEY not set.", {}
        try:
            results   = tavily_client.search(query=args["query"], max_results=4)
            formatted = f"Web search results for: '{args['query']}'\n\n"
            for r in results.get("results", []):
                formatted += f"• {r['title']}\n  {r['url']}\n  {r['content'][:300]}\n\n"
            return formatted, {"type": "web", "query": args["query"]}
        except Exception as e:
            return f"Web search error: {str(e)}", {}

    return "Unknown tool", {}


# ─── Response cache ───────────────────────────────────────────────
def get_cached(query):
    key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    return st.session_state.get("response_cache", {}).get(key)

def set_cached(query, response):
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}
    key   = hashlib.md5(query.lower().strip().encode()).hexdigest()
    cache = st.session_state.response_cache
    cache[key] = response
    if len(cache) > 50:
        del cache[next(iter(cache))]


# ─── Main chat function ───────────────────────────────────────────
def chat(user_message, history, collection):
    """Run one turn. Returns (reply, tool_log, provider_used)."""

    # Check cache
    cached = get_cached(user_message)
    if cached:
        return cached + "\n\n*[cached]*", [], "cache"

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-MAX_HISTORY:]
        + [{"role": "user", "content": user_message}]
    )

    tool_log, provider_used = [], "unknown"

    for _ in range(4):  # max iterations
        try:
            response = call_llm(messages=messages, tools=TOOLS, temperature=0.3)
            provider_used = getattr(response, "_provider_name", "unknown")
        except Exception as e:
            return f"All AI providers are currently unavailable: {str(e)}", [], "error"

        msg = response.choices[0].message

        # No tool call — done
        if not msg.tool_calls:
            reply = msg.content
            set_cached(user_message, reply)
            return reply, tool_log, provider_used

        # Add tool call to messages
        messages.append({
            "role":       "assistant",
            "content":    msg.content,
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        })

        # Execute tools
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            result, meta = execute_tool(tool_name, tool_args, collection)

            tool_log.append({
                "tool":   tool_name,
                "args":   tool_args,
                "result": result[:200] + "..." if len(result) > 200 else result,
                "meta":   meta
            })

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result
            })

    return "Couldn't complete this request. Please try rephrasing.", tool_log, provider_used


# ─── Streamlit UI ─────────────────────────────────────────────────
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

for key, default in [
    ("messages",       []),
    ("llm_history",    []),
    ("ingested",       []),
    ("response_cache", {})
]:
    if key not in st.session_state:
        st.session_state[key] = default

collection = get_collection()

st.title("🤖 AI Assistant")
st.caption("Chat · Upload documents · Search the web — all in one place.")

# Provider status banner
providers = get_providers()
if providers:
    names = " · ".join(f"✅ {p['name']}" for p in providers)
    st.success(f"**Active providers:** {names}", icon="🔌")
else:
    st.error("No API keys found. Add GROQ_API_KEY, GEMINI_API_KEY, or HF_API_KEY.")

sidebar, main = st.columns([0.8, 2.5])

# ── Sidebar ───────────────────────────────────────────────────────
with sidebar:
    st.subheader("📂 Knowledge Base")
    files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if files:
        for f in files:
            if f.name not in st.session_state.ingested:
                with st.spinner(f"Ingesting {f.name}..."):
                    n = ingest_file(f, collection)
                st.success(f"✅ {f.name} — {n} chunks")
                st.session_state.ingested.append(f.name)

    st.divider()
    st.metric("📄 Chunks in KB", collection.count())
    if st.session_state.ingested:
        st.markdown("**Loaded files:**")
        for name in st.session_state.ingested:
            st.markdown(f"▸ {name}")

    st.divider()
    st.subheader("💡 Try asking")
    for s in [
        "What does the document say about X?",
        "Search the web for latest AI news",
        "Summarise everything we've discussed",
        "Compare what the doc says vs current trends",
        "Draft an email based on this document",
    ]:
        if st.button(s, key=s):
            st.session_state["suggestion"] = s

    st.divider()
    st.metric("🗄️ Cached responses", len(st.session_state.response_cache))

    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear chat"):
        st.session_state.messages    = []
        st.session_state.llm_history = []
        st.rerun()
    if col2.button("📄 Clear KB"):
        st.session_state.ingested       = []
        st.session_state.response_cache = {}
        st.cache_resource.clear()
        st.rerun()

# ── Main chat ─────────────────────────────────────────────────────
with main:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools_used"):
                with st.expander("🔧 Tools used"):
                    for t in msg["tools_used"]:
                        icon = "🔍" if t["tool"] == "search_documents" else "🌐"
                        st.markdown(f"{icon} **{t['tool']}** — `{t['args'].get('query', '')}`")
            if msg.get("provider") and msg["provider"] not in ("cache", "unknown", "error"):
                st.caption(f"via {msg['provider']}")

    default_input = st.session_state.pop("suggestion", "")

    if user_input := st.chat_input("Ask anything...") or default_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, tool_log, provider = chat(
                    user_input,
                    st.session_state.llm_history,
                    collection
                )
            st.markdown(reply)

            if tool_log:
                with st.expander("🔧 Tools used"):
                    for t in tool_log:
                        icon = "🔍" if t["tool"] == "search_documents" else "🌐"
                        st.markdown(f"{icon} **{t['tool']}** — `{t['args'].get('query', '')}`")

            if provider == "cache":
                st.caption("⚡ Served from cache")
            elif provider not in ("unknown", "error"):
                st.caption(f"🔌 via {provider}")

        st.session_state.messages.append({
            "role":       "assistant",
            "content":    reply,
            "tools_used": tool_log,
            "provider":   provider
        })
        st.session_state.llm_history.append({"role": "user",      "content": user_input})
        st.session_state.llm_history.append({"role": "assistant", "content": reply})