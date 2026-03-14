import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- System Prompt --- #
SYSTEM_PROMPT = """You are a senior database engineer and SQL expert with 15 years of experience.

When given a SQL query, you always respond in this exact JSON format with no extra text:
{
  "explanation": "Plain English explanation of what this query does, step by step",
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "rewrite": "Optimized version of the query if possible, else write LOOKS GOOD AS IS"
}

Rules:
- explanation must be clear enough for a junior developer to understand
- issues: list actual problems (missing indexes, SELECT *, N+1 risk etc). Empty list [] if none.
- suggestions: practical performance tips. Empty list [] if none.
- rewrite: only rewrite if there is a genuine improvement. Keep original if already optimal.
- Respond in raw JSON only. No markdown. No backticks. No extra text."""


def explain_query(sql: str) -> dict:
    """Send SQL to LLM, get back structured explanation."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,   # low temp — we want consistent structured output
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Explain this SQL query:\n\n{sql}"}
        ]
    )

    raw = response.choices[0].message.content

    # Safe JSON parsing
    import json
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if model adds any extra text
        result = {
            "explanation": raw,
            "issues": [],
            "suggestions": [],
            "rewrite": "Could not parse rewrite."
        }

    return result, response.usage


# --- Streamlit UI --- #
st.set_page_config(
    page_title="SQL Explainer",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ SQL Query Explainer")
st.caption("Paste any SQL query — get a plain English explanation, issues, and an optimized rewrite.")

# Sample queries for quick testing
with st.expander("📋 Load a sample query"):
    samples = {
        "Basic JOIN": """SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'pending';""",
        "Top 5 customers": """SELECT c.name, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY total DESC LIMIT 5;""",
        "Subquery example": """SELECT name FROM employees WHERE department_id IN (SELECT id FROM departments WHERE budget > 100000);""",
        "N+1 risk": """SELECT * FROM posts WHERE user_id IN (SELECT id FROM users WHERE created_at > '2024-01-01');"""
    }

    selected = st.selectbox("Choose a sample", list(samples.keys()))
    if st.button("Load sample"):
        st.session_state["sql_input"] = samples[selected]

# SQL input box
sql_input = st.text_area(
    "Paste your SQL query here",
    value=st.session_state.get("sql_input", ""),
    height=180,
    placeholder="SELECT * FROM orders WHERE status = 'pending'..."
)

# Explain button
if st.button("⚡ Explain Query", type="primary", disabled=not sql_input.strip()):

    with st.spinner("Analysing your query..."):
        result, usage = explain_query(sql_input.strip())

    # Layout — 2 columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📖 What This Query Does")
        st.info(result.get("explanation", "N/A"))

        st.subheader("⚠️ Issues Found")
        issues = result.get("issues", [])
        if issues:
            for issue in issues:
                st.error(f"• {issue}")
        else:
            st.success("No major issues found.")

    with col2:
        st.subheader("💡 Suggestions")
        suggestions = result.get("suggestions", [])
        if suggestions:
            for s in suggestions:
                st.warning(f"• {s}")
        else:
            st.success("Query looks efficient.")

        st.subheader("🔧 Optimized Rewrite")
        rewrite = result.get("rewrite", "")
        if rewrite and rewrite != "LOOKS GOOD AS IS":
            st.code(rewrite, language="sql")
        else:
            st.success("✅ Original query is already optimal.")

    # Token usage at the bottom
    st.divider()
    st.caption(f"🔢 Tokens used — Input: {usage.prompt_tokens} | Output: {usage.completion_tokens} | Total: {usage.total_tokens}")