import os
import json
import time
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── Default Test Suite ──────────────────────────────────────────
# Task: classify customer support tickets into categories
DEFAULT_TEST_CASES = [
    {
        "input": "My order hasn't arrived yet, it's been 2 weeks!",
        "expected": "DELIVERY",
        "id": "TC01"
    },
    {
        "input": "I was charged twice for the same item",
        "expected": "BILLING",
        "id": "TC02"
    },
    {
        "input": "How do I reset my password?",
        "expected": "ACCOUNT",
        "id": "TC03"
    },
    {
        "input": "The product stopped working after 3 days",
        "expected": "PRODUCT",
        "id": "TC04"
    },
    {
        "input": "I want to return this item, it's not what I expected",
        "expected": "RETURNS",
        "id": "TC05"
    },
    {
        "input": "Can I change my delivery address?",
        "expected": "DELIVERY",
        "id": "TC06"
    },
    {
        "input": "My credit card was declined",
        "expected": "BILLING",
        "id": "TC07"
    },
    {
        "input": "I can't log into my account",
        "expected": "ACCOUNT",
        "id": "TC08"
    },
    {
        "input": "This item is broken, I need a replacement",
        "expected": "PRODUCT",
        "id": "TC09"
    },
    {
        "input": "What's your refund policy?",
        "expected": "RETURNS",
        "id": "TC10"
    },
]

# ─── Prompt Versions to Compare ──────────────────────────────────
PROMPT_VERSIONS = {
    "v1 — Basic": """Classify the customer support ticket into one of these categories:
DELIVERY, BILLING, ACCOUNT, PRODUCT, RETURNS
Reply with the category word only.""",

    "v2 — With examples": """Classify the customer support ticket into one of these categories:
DELIVERY, BILLING, ACCOUNT, PRODUCT, RETURNS

Examples:
- "Where is my package?" → DELIVERY
- "I was overcharged" → BILLING
- "I forgot my password" → ACCOUNT
- "Item arrived damaged" → PRODUCT
- "I want a refund" → RETURNS

Reply with the category word only. No explanation.""",

    "v3 — Strict + CoT": """You are a customer support ticket classifier.

Think step by step about what the customer's core problem is, then classify it.

Categories and when to use them:
- DELIVERY: shipping, tracking, lost packages, address changes
- BILLING: charges, payments, invoices, pricing issues
- ACCOUNT: login, password, profile, authentication
- PRODUCT: defects, broken items, quality issues, not working
- RETURNS: refunds, returns, exchanges, policy questions

Your response must be exactly one word from the list above. Nothing else.""",

    "v4 — Minimal": """Support ticket category (DELIVERY/BILLING/ACCOUNT/PRODUCT/RETURNS):""",
}


# ─── Core Functions ───────────────────────────────────────────────

def run_single(prompt: str, user_input: str) -> tuple[str, float]:
    """Run one prompt against one test case. Returns (response, latency_ms)."""
    start = time.time()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,   # deterministic — evals must be reproducible
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
    )
    latency = (time.time() - start) * 1000
    return response.choices[0].message.content.strip(), round(latency, 1)


def score_response(response: str, expected: str) -> dict:
    """Score a response against expected output."""
    response_clean = response.upper().strip()
    expected_clean = expected.upper().strip()

    # Exact match
    if response_clean == expected_clean:
        return {"score": 100, "label": "✅ Exact", "correct": True}

    # Contains match (model added extra words)
    if expected_clean in response_clean:
        return {"score": 70, "label": "⚠️ Partial", "correct": False}

    # Wrong
    return {"score": 0, "label": "❌ Wrong", "correct": False}


def run_eval(prompt_name: str, prompt: str, test_cases: list,
             progress_bar, status_text, col_index: int) -> dict:
    """Run a full eval for one prompt version."""
    results = []
    total_score = 0
    correct = 0
    total_latency = 0

    for i, tc in enumerate(test_cases):
        status_text.text(f"Testing {prompt_name} — case {i+1}/{len(test_cases)}")
        progress_bar.progress((col_index * len(test_cases) + i + 1) /
                              (len(PROMPT_VERSIONS) * len(test_cases)))

        response, latency = run_single(prompt, tc["input"])
        scoring = score_response(response, tc["expected"])

        results.append({
            "id": tc["id"],
            "input": tc["input"],
            "expected": tc["expected"],
            "got": response,
            "score": scoring["score"],
            "label": scoring["label"],
            "correct": scoring["correct"],
            "latency_ms": latency
        })

        total_score += scoring["score"]
        correct += int(scoring["correct"])
        total_latency += latency

        time.sleep(0.1)  # avoid rate limiting

    avg_score = round(total_score / len(test_cases), 1)
    accuracy = round((correct / len(test_cases)) * 100, 1)
    avg_latency = round(total_latency / len(test_cases), 1)

    return {
        "prompt_name": prompt_name,
        "avg_score": avg_score,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "correct": correct,
        "total": len(test_cases),
        "results": results
    }


# ─── Streamlit UI ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Prompt Eval Dashboard",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Prompt Eval Dashboard")
st.caption("Compare prompt versions scientifically. Run all variants against test cases, score automatically, find the winner.")

# ── Sidebar — Config ──
with st.sidebar:
    st.subheader("⚙️ Configuration")

    selected_prompts = st.multiselect(
        "Prompt versions to test",
        options=list(PROMPT_VERSIONS.keys()),
        default=list(PROMPT_VERSIONS.keys())
    )

    st.divider()
    st.subheader("📋 Prompt Viewer")
    view_prompt = st.selectbox("Preview a prompt", list(PROMPT_VERSIONS.keys()))
    st.code(PROMPT_VERSIONS[view_prompt], language="text")

# ── Main area ──
st.subheader("📝 Test Cases")
with st.expander(f"View all {len(DEFAULT_TEST_CASES)} test cases"):
    for tc in DEFAULT_TEST_CASES:
        st.markdown(f"**{tc['id']}** `{tc['expected']}` — {tc['input']}")

st.divider()

# ── Run button ──
if st.button("🚀 Run Eval", type="primary",
             disabled=len(selected_prompts) == 0):

    all_results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, prompt_name in enumerate(selected_prompts):
        prompt = PROMPT_VERSIONS[prompt_name]
        result = run_eval(
            prompt_name, prompt, DEFAULT_TEST_CASES,
            progress_bar, status_text, i
        )
        all_results[prompt_name] = result

    progress_bar.progress(1.0)
    status_text.text("✅ Eval complete!")

    # ── Leaderboard ──
    st.subheader("🏆 Leaderboard")

    sorted_results = sorted(
        all_results.values(),
        key=lambda x: (x["accuracy"], -x["avg_latency_ms"]),
        reverse=True
    )

    for rank, res in enumerate(sorted_results):
        medal = ["🥇", "🥈", "🥉", "4️⃣"][rank] if rank < 4 else f"{rank+1}."
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        col1.markdown(f"{medal} **{res['prompt_name']}**")
        col2.metric("Accuracy", f"{res['accuracy']}%")
        col3.metric("Avg Score", f"{res['avg_score']}")
        col4.metric("Correct", f"{res['correct']}/{res['total']}")
        col5.metric("Avg Latency", f"{res['avg_latency_ms']}ms")

    st.divider()

    # ── Detailed results per prompt ──
    st.subheader("🔬 Detailed Results")

    tabs = st.tabs([r["prompt_name"] for r in sorted_results])

    for tab, res in zip(tabs, sorted_results):
        with tab:
            for r in res["results"]:
                color = "green" if r["correct"] else "red"
                with st.expander(
                    f"{r['label']} {r['id']} — Expected: `{r['expected']}` | Got: `{r['got']}` | {r['latency_ms']}ms"
                ):
                    st.markdown(f"**Input:** {r['input']}")
                    st.markdown(f"**Expected:** `{r['expected']}`")
                    st.markdown(f"**Got:** `{r['got']}`")
                    st.markdown(f"**Score:** {r['score']}/100")

    # ── Save results ──
    st.divider()
    results_json = json.dumps(all_results, indent=2)
    st.download_button(
        "💾 Download Results JSON",
        data=results_json,
        file_name="eval_results.json",
        mime="application/json"
    )