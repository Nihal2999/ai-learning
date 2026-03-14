import os
import hashlib
import streamlit as st
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── Config ──────────────────────────────────────────────────────
CHUNK_SIZE     = 500    # characters per chunk
CHUNK_OVERLAP  = 100    # overlap between chunks
TOP_K          = 4      # how many chunks to retrieve per query
DB_PATH        = "./chroma_db"  # local vector DB storage

# ─── Embedding model (runs locally) ──────────────────────────────
@st.cache_resource
def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

# ─── ChromaDB client ─────────────────────────────────────────────
# @st.cache_resource
# def get_chroma_collection():
#     chroma_client = chromadb.PersistentClient(path=DB_PATH)
#     return chroma_client.get_or_create_collection(
#         name="knowledge_base",
#         embedding_function=get_embedding_fn()
#     )

@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.EphemeralClient()  # in-memory, no disk
    return chroma_client.get_or_create_collection(
        name="knowledge_base",
        embedding_function=get_embedding_fn()
    )

# ─── Text extraction ─────────────────────────────────────────────
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(file) -> str:
    return file.read().decode("utf-8")

# ─── Chunking ────────────────────────────────────────────────────
def chunk_text(text: str, source_name: str) -> list[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:  # skip empty chunks
            chunks.append({
                "text": chunk,
                "id": f"{source_name}_chunk_{chunk_index}",
                "source": source_name,
                "chunk_index": chunk_index
            })
            chunk_index += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP  # overlap

    return chunks

# ─── Ingestion ───────────────────────────────────────────────────
def ingest_document(file, collection) -> int:
    """Extract, chunk, embed and store a document."""
    name = file.name

    # Extract text
    if name.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    else:
        text = extract_text_from_txt(file)

    if not text.strip():
        return 0

    # Chunk it
    chunks = chunk_text(text, name)

    # Store in ChromaDB
    collection.upsert(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[{"source": c["source"],
                    "chunk_index": c["chunk_index"]} for c in chunks]
    )

    return len(chunks)

# ─── Retrieval ───────────────────────────────────────────────────
def retrieve(query: str, collection, top_k: int = TOP_K) -> list[dict]:
    """Find the most relevant chunks for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count())
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i]
        })

    return chunks

# ─── Generation ──────────────────────────────────────────────────
def generate_answer(query: str, chunks: list[dict]) -> tuple[str, int]:
    """Generate an answer using retrieved chunks as context."""

    # Build context block
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n"

    system_prompt = """You are a helpful assistant that answers questions
based ONLY on the provided context.

Rules:
- Only use information from the provided context
- Always cite which source you used, e.g. (Source 1)
- If the answer is not in the context, say "I couldn't find that in the provided documents"
- Be concise and accurate"""

    user_message = f"""Context:
{context}

Question: {query}

Answer based only on the context above:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return (
        response.choices[0].message.content,
        response.usage.total_tokens
    )

# ─── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Base Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Personal Knowledge Base Q&A")
st.caption("Upload documents → ask questions → get answers with source citations.")

collection = get_chroma_collection()

# ── Sidebar — Upload & Stats ──
with st.sidebar:
    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"Ingesting {file.name}..."):
                count = ingest_document(file, collection)
            if count:
                st.success(f"✅ {file.name} → {count} chunks")

    st.divider()
    st.subheader("📊 Knowledge Base Stats")
    total = collection.count()
    st.metric("Total chunks stored", total)

    if total > 0:
        if st.button("🗑️ Clear Knowledge Base"):
            # Delete and recreate collection
            # chroma_client = chromadb.PersistentClient(path=DB_PATH)
            # chroma_client.delete_collection("knowledge_base")
            st.cache_resource.clear()
            st.rerun()

# ── Main — Q&A ──
if collection.count() == 0:
    st.info("👈 Upload some documents in the sidebar to get started.")
    st.markdown("""
    **What you can upload:**
    - PDF files (books, papers, documentation)
    - Text files (.txt)

    **Then ask anything like:**
    - "What is the main argument of chapter 2?"
    - "Summarise the key points about X"
    - "What does the document say about Y?"
    """)
else:
    st.subheader("💬 Ask a Question")

    query = st.text_input(
        "Your question",
        placeholder="What does the document say about...?"
    )

    col1, col2 = st.columns([1, 4])
    top_k = col1.slider("Sources to retrieve", 1, 8, TOP_K)

    if st.button("🔍 Search & Answer", type="primary",
                 disabled=not query.strip()):

        with st.spinner("Searching knowledge base..."):
            chunks = retrieve(query, collection, top_k)

        with st.spinner("Generating answer..."):
            answer, tokens = generate_answer(query, chunks)

        # ── Answer ──
        st.subheader("✅ Answer")
        st.markdown(answer)

        # ── Retrieved chunks ──
        st.divider()
        st.subheader(f"📎 Retrieved Sources ({len(chunks)} chunks)")

        for i, chunk in enumerate(chunks):
            similarity = round((1 - chunk["distance"]) * 100, 1)
            with st.expander(
                f"Source {i+1} — {chunk['source']} "
                f"(chunk #{chunk['chunk_index']}, "
                f"similarity: {similarity}%)"
            ):
                st.markdown(chunk["text"])

        st.divider()
        st.caption(f"🔢 Total tokens used: {tokens}")