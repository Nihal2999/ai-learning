import os
import json
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ─── API Key ─────────────────────────────────────────────────────
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

# ─── Config ──────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"

CATEGORIES = [
    "URGENT",
    "BILLING",
    "SUPPORT",
    "SALES",
    "HR",
    "LEGAL",
    "SPAM",
    "INFO"
]

SYSTEM_PROMPT = """You are an expert email triage assistant for a software company.

Analyse every email and respond in this exact JSON format with no extra text:
{
  "category": "one of: URGENT, BILLING, SUPPORT, SALES, HR, LEGAL, SPAM, INFO",
  "urgency": "HIGH, MEDIUM, or LOW",
  "sender_intent": "One sentence: what does the sender want?",
  "key_asks": ["ask 1", "ask 2"],
  "deadlines": ["any deadlines mentioned, or empty list"],
  "sentiment": "POSITIVE, NEUTRAL, NEGATIVE, or ANGRY",
  "should_escalate": true or false,
  "escalation_reason": "why escalate, or empty string if not",
  "suggested_reply": "A professional, concise reply to this email. 3-5 sentences.",
  "summary": "One sentence summary of the email"
}

Rules:
- URGENT: anything time-sensitive, system down, legal threat, angry customer
- BILLING: invoices, payments, refunds, pricing questions
- SUPPORT: technical help, bugs, how-to questions
- SALES: new business, partnerships, vendor pitches
- HR: hiring, benefits, leave, internal team matters
- LEGAL: contracts, compliance, NDAs, legal threats
- SPAM: marketing, unsolicited, irrelevant
- INFO: FYI emails, newsletters, general updates
- should_escalate = true if: HIGH urgency + NEGATIVE/ANGRY sentiment, or LEGAL category, or URGENT category
- suggested_reply must be professional and directly address the sender's key asks"""


# ─── Core functions ───────────────────────────────────────────────
def triage_email(subject: str, body: str, sender: str = "") -> dict:
    """Run triage on a single email."""

    email_text = f"""From: {sender if sender else 'Unknown'}
Subject: {subject}
Body:
{body}"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Triage this email:\n\n{email_text}"}
        ]
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip("```json").strip("```").strip()
        result = json.loads(cleaned)

    result["subject"]    = subject
    result["sender"]     = sender
    result["body"]       = body
    result["triaged_at"] = datetime.now().strftime("%H:%M:%S")
    result["tokens"]     = response.usage.total_tokens

    return result


def get_urgency_color(urgency: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(urgency, "⚪")


def get_category_emoji(category: str) -> str:
    return {
        "URGENT":  "🚨",
        "BILLING": "💰",
        "SUPPORT": "🔧",
        "SALES":   "📈",
        "HR":      "👥",
        "LEGAL":   "⚖️",
        "SPAM":    "🗑️",
        "INFO":    "ℹ️"
    }.get(category, "📧")


# ─── Sample emails for testing ────────────────────────────────────
SAMPLE_EMAILS = {
    "Angry customer — refund": {
        "sender": "angry.customer@gmail.com",
        "subject": "UNACCEPTABLE - Charged twice, no response for 2 weeks!!",
        "body": """I have been charged TWICE for my subscription and despite sending 3 emails over the past 2 weeks, nobody has responded. This is completely unacceptable.

I want a full refund of both charges immediately. If this is not resolved within 24 hours I will be disputing the charges with my bank and leaving a 1-star review on every platform I can find.

My order ID is #45821. Amount charged: ₹4,999 x 2 = ₹9,998.

This is your last chance to make this right."""
    },
    "Server down alert": {
        "sender": "monitoring@company.com",
        "subject": "CRITICAL: Production API server down - 500 errors",
        "body": """Alert triggered at 14:32 IST.

Production API server api.company.com is returning 500 errors for 100% of requests.
Affected services: Payment processing, User authentication, Data sync.
Error rate: 847 errors/minute.
Duration: 8 minutes and counting.

On-call engineer has been notified but requires immediate team lead attention.
Estimated revenue impact: ₹2,000/minute."""
    },
    "Job application": {
        "sender": "candidate@gmail.com",
        "subject": "Application for Senior Python Developer position",
        "body": """Dear Hiring Team,

I am writing to express my interest in the Senior Python Developer position posted on LinkedIn.

I have 5 years of experience with Python, FastAPI, and PostgreSQL. I have worked on several high-scale backend systems and am passionate about clean, maintainable code.

I have attached my resume and portfolio for your review. I am available for an interview at your convenience.

Looking forward to hearing from you.

Best regards,
Rahul Sharma"""
    },
    "Invoice due": {
        "sender": "accounts@vendor.com",
        "subject": "Invoice #INV-2024-891 Due in 3 days",
        "body": """Dear Team,

Please find attached Invoice #INV-2024-891 for ₹45,000 for cloud infrastructure services for March 2026.

Payment is due by March 18, 2026.

Bank details:
Account: HDFC Bank
Account No: XXXX-XXXX-4521

Please confirm receipt and expected payment date.

Regards,
Accounts Team - CloudVendor Ltd"""
    },
    "Feature request": {
        "sender": "user123@startup.com",
        "subject": "Feature request: bulk export functionality",
        "body": """Hi Support Team,

Love your product! One thing that would really help our workflow is the ability to bulk export data to CSV.

Currently we have to export records one by one which takes forever with 500+ entries.

Is this something on your roadmap? Happy to hop on a call to discuss our use case.

Thanks,
Priya"""
    },
    "Legal notice": {
        "sender": "legal@lawfirm.com",
        "subject": "Notice of Intellectual Property Infringement",
        "body": """Dear Sir/Madam,

We represent XYZ Corporation in matters of intellectual property.

It has come to our attention that your product incorporates design elements and functionality that closely resembles our client's patented technology (Patent No. IN-2019-04521).

We formally request that you cease and desist from further use of these elements within 7 days of this notice, failing which our client reserves the right to pursue legal action.

Please have your legal counsel contact us within 48 hours.

Regards,
Advocate Suresh Menon
Senior Partner, Menon & Associates"""
    }
}


# ─── Streamlit UI ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Triage Agent",
    page_icon="📧",
    layout="wide"
)

# Init session state
if "triage_history" not in st.session_state:
    st.session_state.triage_history = []

st.title("📧 Email Triage Agent")
st.caption("Paste any email → AI classifies, extracts info, drafts reply, flags urgent.")

# ── Layout: input left, dashboard right ──
left, right = st.columns([1.2, 1])

with left:
    st.subheader("📨 New Email")

    # Sample loader
    with st.expander("📋 Load a sample email"):
        selected = st.selectbox("Choose sample", list(SAMPLE_EMAILS.keys()))
        if st.button("Load sample"):
            sample = SAMPLE_EMAILS[selected]
            st.session_state["sender"]  = sample["sender"]
            st.session_state["subject"] = sample["subject"]
            st.session_state["body"]    = sample["body"]

    sender  = st.text_input("From",    value=st.session_state.get("sender", ""),  placeholder="sender@example.com")
    subject = st.text_input("Subject", value=st.session_state.get("subject", ""), placeholder="Email subject line")
    body    = st.text_area("Body",     value=st.session_state.get("body", ""),    height=280, placeholder="Paste email body here...")

    triage_btn = st.button(
        "⚡ Triage Email",
        type="primary",
        disabled=not (subject.strip() and body.strip())
    )

    if triage_btn:
        with st.spinner("Agent analysing email..."):
            result = triage_email(subject, body, sender)
            st.session_state.triage_history.insert(0, result)

        # ── Result display ──
        st.divider()

        # Header row
        urg   = get_urgency_color(result["urgency"])
        cat   = get_category_emoji(result["category"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Category",  f"{cat} {result['category']}")
        col2.metric("Urgency",   f"{urg} {result['urgency']}")
        col3.metric("Sentiment", result["sentiment"])

        # Escalation alert
        if result.get("should_escalate"):
            st.error(f"🚨 **ESCALATE:** {result['escalation_reason']}")

        # Summary
        st.info(f"**Summary:** {result['summary']}")

        # Two columns — intent/asks/deadlines | reply
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**🎯 Sender Intent**")
            st.markdown(result["sender_intent"])

            if result["key_asks"]:
                st.markdown("**📋 Key Asks**")
                for ask in result["key_asks"]:
                    st.markdown(f"▸ {ask}")

            if result["deadlines"]:
                st.warning("**⏰ Deadlines**")
                for d in result["deadlines"]:
                    st.markdown(f"▸ {d}")

        with c2:
            st.markdown("**✍️ Suggested Reply**")
            st.text_area(
                "Copy this reply",
                value=result["suggested_reply"],
                height=200,
                key=f"reply_{len(st.session_state.triage_history)}"
            )

        st.caption(f"🔢 Tokens used: {result['tokens']}")

with right:
    st.subheader("📊 Triage Dashboard")

    history = st.session_state.triage_history

    if not history:
        st.info("No emails triaged yet. Process an email to see the dashboard.")
    else:
        # Summary metrics
        total    = len(history)
        urgent   = sum(1 for e in history if e["urgency"] == "HIGH")
        escalate = sum(1 for e in history if e.get("should_escalate"))
        tokens   = sum(e.get("tokens", 0) for e in history)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",    total)
        m2.metric("🔴 High",  urgent)
        m3.metric("🚨 Escalate", escalate)
        m4.metric("Tokens",   tokens)

        st.divider()

        # Category breakdown
        from collections import Counter
        cats = Counter(e["category"] for e in history)
        st.markdown("**By Category**")
        for cat_name, count in cats.most_common():
            emoji = get_category_emoji(cat_name)
            bar   = "█" * count
            st.markdown(f"`{emoji} {cat_name}` {bar} {count}")

        st.divider()

        # Email list
        st.markdown("**Processed Emails**")
        for i, email in enumerate(history):
            urg_icon = get_urgency_color(email["urgency"])
            cat_icon = get_category_emoji(email["category"])
            label    = f"{urg_icon} {cat_icon} {email['subject'][:40]}..."

            with st.expander(label):
                st.markdown(f"**From:** {email['sender']}")
                st.markdown(f"**Category:** {email['category']} | **Urgency:** {email['urgency']}")
                st.markdown(f"**Summary:** {email['summary']}")
                if email.get("should_escalate"):
                    st.error(f"🚨 {email['escalation_reason']}")
                st.markdown(f"**Reply:** {email['suggested_reply']}")
                st.caption(f"Triaged at {email['triaged_at']}")

        # Clear button
        st.divider()
        if st.button("🗑️ Clear history"):
            st.session_state.triage_history = []
            st.rerun()