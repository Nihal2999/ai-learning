import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── System Prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert data analyst and Python/pandas engineer.

The user will give you a question about a dataset. You will be given:
- The column names and data types of the dataset
- A few sample rows
- The user's question

You must respond in this exact JSON format with no extra text, no markdown, no backticks:
{
  "thought": "Your reasoning about how to answer this question",
  "code": "Single line or multiline pandas Python code. Use 'df' as the dataframe variable. Store the final result in a variable called 'result'. result must always be a DataFrame or a scalar value.",
  "chart_type": "bar | line | pie | scatter | none",
  "chart_x": "column name for x axis, or null",
  "chart_y": "column name for y axis, or null",
  "summary": "Plain English answer to the user's question in 1-2 sentences"
}

Rules:
- Always use 'df' as the dataframe name
- Always store final output in 'result'
- For aggregations, result should be a DataFrame with meaningful column names
- chart_type must be one of: bar, line, pie, scatter, none
- If no chart makes sense, set chart_type to none
- Never use print() statements
- Never import anything — pandas is already imported as pd"""


def get_dataset_context(df: pd.DataFrame) -> str:
    """Build a description of the dataframe to send to the LLM."""
    context = f"Columns and types:\n"
    for col, dtype in df.dtypes.items():
        context += f"  - {col}: {dtype}\n"
    context += f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns\n"
    context += f"\nSample rows (first 3):\n{df.head(3).to_string()}"
    return context


def ask_llm(df: pd.DataFrame, question: str) -> dict:
    """Send question + dataset context to LLM, get back analysis plan."""
    context = get_dataset_context(df)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Dataset info:\n{context}\n\nQuestion: {question}"}
        ]
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw), response.usage
    except json.JSONDecodeError:
        # Strip any accidental markdown fences
        cleaned = raw.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned), response.usage


def execute_code(df: pd.DataFrame, code: str):
    """Safely execute LLM-generated pandas code."""
    local_vars = {"df": df, "pd": pd}
    try:
        exec(code, {}, local_vars)
        return local_vars.get("result", None), None
    except Exception as e:
        return None, str(e)


def render_chart(result: pd.DataFrame, chart_type: str, x: str, y: str):
    """Render a plotly chart based on LLM's suggestion."""
    try:
        if chart_type == "bar":
            fig = px.bar(result, x=x, y=y)
        elif chart_type == "line":
            fig = px.line(result, x=x, y=y)
        elif chart_type == "pie":
            fig = px.pie(result, names=x, values=y)
        elif chart_type == "scatter":
            fig = px.scatter(result, x=x, y=y)
        else:
            return
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart could not render: {e}")


# ─── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analyst Agent")
st.caption("Upload any CSV → ask questions in plain English → get answers + charts.")

# ── File Upload ──
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing values", df.isnull().sum().sum())

    st.divider()

    # ── Sample Questions ──
    st.subheader("💬 Ask a Question")

    sample_questions = [
        "What are the top 5 values by count?",
        "Show me the distribution of the first numeric column",
        "What is the average of each numeric column?",
        "Are there any missing values? Show me which columns",
        "Show monthly trends if there is a date column",
    ]

    with st.expander("💡 Sample questions"):
        for q in sample_questions:
            if st.button(q, key=q):
                st.session_state["question"] = q

    question = st.text_input(
        "Your question",
        value=st.session_state.get("question", ""),
        placeholder="e.g. What are the top 5 customers by total sales?"
    )

    if st.button("🔍 Analyse", type="primary", disabled=not question.strip()):

        with st.spinner("Agent is thinking..."):
            try:
                plan, usage = ask_llm(df, question)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()

        # ── Show agent's reasoning ──
        with st.expander("🧠 Agent's reasoning"):
            st.write(plan.get("thought", ""))
            st.code(plan.get("code", ""), language="python")

        # ── Execute the generated code ──
        result, error = execute_code(df, plan.get("code", ""))

        if error:
            st.error(f"Code execution error: {error}")
        else:
            # ── Answer ──
            st.subheader("✅ Answer")
            st.success(plan.get("summary", ""))

            # ── Result table ──
            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
            elif result is not None:
                st.metric("Result", result)

            # ── Chart ──
            chart_type = plan.get("chart_type", "none")
            if chart_type != "none" and isinstance(result, pd.DataFrame):
                st.subheader("📈 Chart")
                render_chart(
                    result,
                    chart_type,
                    plan.get("chart_x"),
                    plan.get("chart_y")
                )

            # ── Token usage ──
            st.divider()
            st.caption(f"🔢 Tokens — Input: {usage.prompt_tokens} | Output: {usage.completion_tokens} | Total: {usage.total_tokens}")

else:
    # ── Empty state ──
    st.info("👆 Upload a CSV file to get started.")

    st.subheader("What can you ask?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Aggregations**
        - "What are total sales by region?"
        - "Which product has the highest average rating?"
        - "Show me revenue by month"
        """)
    with col2:
        st.markdown("""
        **Exploration**
        - "How many rows have missing values?"
        - "What is the distribution of customer ages?"
        - "Show top 10 by order value"
        """)