import os
import json
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────
try:
    groq_key   = st.secrets["GROQ_API_KEY"]
    tavily_key = st.secrets["TAVILY_API_KEY"]
except:
    groq_key   = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

groq_client   = Groq(api_key=groq_key)
tavily_client = TavilyClient(api_key=tavily_key)

# ─── Config ──────────────────────────────────────────────────────
MODEL         = "llama-3.3-70b-versatile"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 3
MAX_HISTORY   = 10

# ─── Tools available to the assistant ────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the uploaded knowledge base documents for relevant information. Use this when the user asks about something that might be in their uploaded files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant document chunks"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information, news, or facts not in the documents. Use this for recent events, current data, or topics not covered in uploaded files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query"
                    }
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
- For questions about uploaded documents → use search_documents
- For current events, recent news, or topics not in documents → use search_web
- For general knowledge questions → answer directly without tools
- Always cite your sources (document name or web URL)
- Be concise and helpful
- Remember the full conversation history"""


# ─── Embedding + ChromaDB ────────────────────────────────────────
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

# ─── Document ingestion ──────────────────────────────────────────
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

# ─── Tool execution ──────────────────────────────────────────────
def execute_tool(name: str, args: dict, collection) -> tuple[str, dict]:
    """Execute a tool and return (result_text, metadata)."""

    if name == "search_documents":
        if collection.count() == 0:
            return "No documents uploaded yet.", {}

        results = collection.query(
            query_texts=[args["query"]],
            n_results=min(TOP_K, collection.count())
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            source = results["metadatas"][0][i]["source"]
            text   = results["documents"][0][i]
            sim    = round((1 - results["distances"][0][i]) * 100, 1)
            chunks.append(f"[{source} | {sim}% match]\n{text}")

        result = "\n\n".join(chunks) if chunks else "No relevant documents found."
        meta   = {"type": "document", "query": args["query"], "chunks": len(chunks)}
        return result, meta

    elif name == "search_web":
        try:
            results   = tavily_client.search(query=args["query"], max_results=4)
            formatted = f"Web search results for: '{args['query']}'\n\n"
            for r in results.get("results", []):
                formatted += f"• {r['title']}\n  {r['url']}\n  {r['content'][:300]}\n\n"
            meta = {"type": "web", "query": args["query"]}
            return formatted, meta
        except Exception as e:
            return f"Web search error: {str(e)}", {}

    return "Unknown tool", {}

# ─── Main chat function ───────────────────────────────────────────
def chat(
    user_message: str,
    history: list,
    collection
) -> tuple[str, list]:

    # Build messages — trim history more aggressively for tool calls
    trimmed_history = history[-6:]  # keep last 3 turns only

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + trimmed_history
        + [{"role": "user", "content": user_message}]
    )

    tool_log = []
    max_iterations = 4

    for _ in range(max_iterations):
        try:
            response = groq_client.chat.completions.create(
                model=MODEL,
                temperature=0.3,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
        except Exception as e:
            # If tool calling fails due to long context — retry without tools
            if "tool_use_failed" in str(e) or "400" in str(e):
                try:
                    response = groq_client.chat.completions.create(
                        model=MODEL,
                        temperature=0.3,
                        messages=messages
                        # No tools — plain completion fallback
                    )
                    return response.choices[0].message.content, tool_log
                except Exception as e2:
                    return f"I encountered an error: {str(e2)}", tool_log
            return f"I encountered an error: {str(e)}", tool_log

        msg = response.choices[0].message

        # No tool call — final answer
        if not msg.tool_calls:
            return msg.content, tool_log

        # Add assistant tool call to messages
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

        # Execute each tool
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

    return "I wasn't able to complete this request. Please try rephrasing.", tool_log

# ─── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Init session state
if "messages"    not in st.session_state:
    st.session_state.messages    = []
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []
if "ingested"    not in st.session_state:
    st.session_state.ingested    = []

collection = get_collection()

# ── Layout ────────────────────────────────────────────────────────
st.title("🤖 AI Assistant")
st.caption("Chat · Upload documents · Search the web — all in one place.")

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
    suggestions = [
        "What does the document say about X?",
        "Search the web for latest AI news",
        "Summarise everything we've discussed",
        "Compare what the doc says vs current trends",
        "Draft an email based on this document",
    ]
    for s in suggestions:
        if st.button(s, key=s):
            st.session_state["suggestion"] = s

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear chat"):
        st.session_state.messages    = []
        st.session_state.llm_history = []
        st.rerun()
    if col2.button("📄 Clear KB"):
        st.session_state.ingested = []
        st.cache_resource.clear()
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────
with main:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools_used"):
                with st.expander("🔧 Tools used"):
                    for t in msg["tools_used"]:
                        icon = "🔍" if t["tool"] == "search_documents" else "🌐"
                        st.markdown(f"{icon} **{t['tool']}** — `{t['args']}`")

    # Suggestion handling
    default_input = st.session_state.pop("suggestion", "")

    # Chat input
    if user_input := st.chat_input(
        "Ask anything — I can search your documents or the web...",
        key="chat_input"
    ) or default_input:

        # Show user message
        st.session_state.messages.append({
            "role":    "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, tool_log = chat(
                    user_input,
                    st.session_state.llm_history,
                    collection
                )

            st.markdown(reply)

            if tool_log:
                with st.expander("🔧 Tools used"):
                    for t in tool_log:
                        icon = "🔍" if t["tool"] == "search_documents" else "🌐"
                        st.markdown(
                            f"{icon} **{t['tool']}** — `{t['args'].get('query', '')}`"
                        )

        # Save to history
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    reply,
            "tools_used": tool_log
        })
        st.session_state.llm_history.append({
            "role": "user", "content": user_input
        })
        st.session_state.llm_history.append({
            "role": "assistant", "content": reply
        })