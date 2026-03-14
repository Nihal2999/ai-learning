import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from groq import Groq
import streamlit as st

try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=api_key)

# ─── Config ──────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 3
DB_PATH       = "./policy_chroma_db"
MAX_HISTORY   = 6  # last 6 turns kept in memory

# ─── Embedding + ChromaDB ────────────────────────────────────────
@st.cache_resource
def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name="policy_docs",
        embedding_function=ef
    )

# ─── Ingestion ───────────────────────────────────────────────────
def extract_text(file) -> str:
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(p.extract_text() for p in reader.pages)
    return file.read().decode("utf-8")

def ingest(file, collection) -> int:
    text = extract_text(file)
    chunks, start, idx = [], 0, 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append({
                "id": f"{file.name}_c{idx}",
                "text": chunk,
                "source": file.name
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    collection.upsert(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks]
    )
    return len(chunks)

# ─── Retrieval ───────────────────────────────────────────────────
def retrieve(query: str, collection) -> list[dict]:
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(TOP_K, collection.count())
    )
    return [
        {
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "similarity": round((1 - results["distances"][0][i]) * 100, 1)
        }
        for i in range(len(results["documents"][0]))
    ]

# ─── Conversational RAG ──────────────────────────────────────────
def chat_with_rag(
    user_query: str,
    chat_history: list,
    collection
) -> tuple[str, list[dict], int]:

    # 1. Retrieve relevant chunks for THIS query
    chunks = retrieve(user_query, collection)

    # 2. Build context from retrieved chunks
    context = "\n\n".join([
        f"[{c['source']} | {c['similarity']}% match]\n{c['text']}"
        for c in chunks
    ])

    # 3. System prompt
    system = """You are a helpful assistant for company policy and documentation.

Answer questions using ONLY the provided context.
Always cite your source file name.
If the answer isn't in the context, say so clearly.
Keep answers concise and professional."""

    # 4. Build messages — system + trimmed history + context + new question
    context_message = f"""Context from knowledge base:
{context}

---
Question: {user_query}"""

    messages = (
        [{"role": "system", "content": system}]
        + chat_history[-MAX_HISTORY:]        # trimmed history
        + [{"role": "user", "content": context_message}]
    )

    # 5. Generate
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=messages
    )

    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens

    return answer, chunks, tokens


# ─── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(
    page_title="Policy Chatbot",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Company Policy Chatbot")
st.caption("Upload policy documents → have a full conversation → answers grounded in your docs.")

collection = get_collection()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []        # displayed chat
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []     # raw history sent to LLM

# ── Sidebar ──
with st.sidebar:
    st.subheader("📂 Upload Documents")
    files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if files:
        for f in files:
            with st.spinner(f"Ingesting {f.name}..."):
                n = ingest(f, collection)
            st.success(f"✅ {f.name} — {n} chunks")

    st.divider()
    st.metric("Chunks in KB", collection.count())

    if st.button("🗑️ Clear all"):
        c = chromadb.PersistentClient(path=DB_PATH)
        c.delete_collection("policy_docs")
        st.session_state.messages = []
        st.session_state.llm_history = []
        st.cache_resource.clear()
        st.rerun()

    if st.button("🧹 Clear chat only"):
        st.session_state.messages = []
        st.session_state.llm_history = []
        st.rerun()

# ── Chat UI ──
if collection.count() == 0:
    st.info("👈 Upload documents in the sidebar to begin.")
else:
    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources used"):
                    for s in msg["sources"]:
                        st.caption(
                            f"• {s['source']} — {s['similarity']}% match"
                        )

    # Chat input
    if user_input := st.chat_input("Ask about your documents..."):

        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching docs and thinking..."):
                answer, sources, tokens = chat_with_rag(
                    user_input,
                    st.session_state.llm_history,
                    collection
                )

            st.markdown(answer)

            if sources:
                with st.expander("📎 Sources used"):
                    for s in sources:
                        st.caption(
                            f"• {s['source']} — {s['similarity']}% match"
                        )

            st.caption(f"🔢 {tokens} tokens")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        # Update LLM history (plain role/content only)
        st.session_state.llm_history.append({
            "role": "user", "content": user_input
        })
        st.session_state.llm_history.append({
            "role": "assistant", "content": answer
        })