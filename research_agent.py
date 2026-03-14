import os
import json
import streamlit as st
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ─── Clients ─────────────────────────────────────────────────────
try:
    groq_client  = Groq(api_key=st.secrets["GROQ_API_KEY"])
    tavily       = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
except:
    groq_client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
    tavily       = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ─── Config ──────────────────────────────────────────────────────
MAX_ITERATIONS = 6    # max agent steps before forcing conclusion
MAX_SEARCHES   = 4    # max web searches per run
MODEL          = "llama-3.3-70b-versatile"

# ─── Tool definitions ────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for recent information on a topic. Returns titles, URLs and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch and read the full content of a webpage URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": "Write the final research report when you have gathered enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Report title"
                    },
                    "executive_summary": {
                        "type": "string",
                        "description": "2-3 sentence summary of key findings"
                    },
                    "key_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of 4-6 key findings with source references"
                    },
                    "perspectives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-3 different viewpoints or perspectives on the topic"
                    },
                    "conclusion": {
                        "type": "string",
                        "description": "Concluding paragraph"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs used as sources"
                    }
                },
                "required": ["title", "executive_summary", "key_findings", "conclusion", "sources"]
            }
        }
    }
]

# ─── Tool execution ───────────────────────────────────────────────
def execute_tool(name: str, args: dict, search_count: list) -> str:
    """Execute a tool and return the result as a string."""

    if name == "search_web":
        if search_count[0] >= MAX_SEARCHES:
            return "Search limit reached. Please write the report now."
        try:
            results = tavily.search(
                query=args["query"],
                max_results=5,
                search_depth="basic"
            )
            search_count[0] += 1
            formatted = f"Search results for: '{args['query']}'\n\n"
            for i, r in enumerate(results.get("results", []), 1):
                formatted += f"{i}. {r['title']}\n"
                formatted += f"   URL: {r['url']}\n"
                formatted += f"   {r['content'][:300]}...\n\n"
            return formatted
        except Exception as e:
            return f"Search error: {str(e)}"

    elif name == "fetch_page":
        try:
            result = tavily.extract(urls=[args["url"]])
            content = result["results"][0]["raw_content"][:3000]
            return f"Content from {args['url']}:\n\n{content}"
        except Exception as e:
            return f"Could not fetch page: {str(e)}"

    elif name == "write_report":
        return json.dumps(args)

    return "Unknown tool"


# ─── Agent loop ───────────────────────────────────────────────────
def run_agent(topic: str, progress_container, log_container):
    """Run the research agent and return the final report."""

    system_prompt = f"""You are an expert research agent. Today's date is {datetime.now().strftime('%B %d, %Y')}.

Your job is to research a topic thoroughly and write a comprehensive report.

Process:
1. Search for the topic using multiple different search queries
2. Read the most relevant pages in full
3. Gather diverse perspectives and recent data
4. Write a structured report using write_report

Rules:
- Use at least 3 different search queries before writing the report
- Always fetch at least 1 page in full for deeper context
- Include specific facts, numbers, and quotes where possible
- Always cite your sources in findings
- Write the report only when you have enough information"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Research this topic thoroughly: {topic}"}
    ]

    search_count = [0]
    iteration    = 0
    agent_log    = []
    final_report = None

    while iteration < MAX_ITERATIONS:
        iteration += 1
        progress_container.progress(iteration / MAX_ITERATIONS)

        # Call LLM with tools
        response = groq_client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        # No tool call = agent is done
        if not msg.tool_calls:
            agent_log.append({
                "step": iteration,
                "type": "final_text",
                "content": msg.content
            })
            break

        # Process tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        })

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)

            # Log the action
            log_entry = {
                "step": iteration,
                "type": tool_name,
                "args": tool_args
            }

            # Execute
            result = execute_tool(tool_name, tool_args, search_count)

            # If write_report called — extract and store
            if tool_name == "write_report":
                try:
                    final_report = json.loads(result)
                    log_entry["result"] = "Report written"
                except:
                    log_entry["result"] = result
            else:
                log_entry["result"] = result[:200] + "..." if len(result) > 200 else result

            agent_log.append(log_entry)

            # Update live log
            with log_container:
                for entry in agent_log:
                    if entry["type"] == "search_web":
                        st.markdown(f"🔍 **Searched:** `{entry['args'].get('query', '')}`")
                    elif entry["type"] == "fetch_page":
                        st.markdown(f"📄 **Read page:** `{entry['args'].get('url', '')[:60]}...`")
                    elif entry["type"] == "write_report":
                        st.markdown(f"✍️ **Writing report...**")

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

            # Stop if report written
            if tool_name == "write_report" and final_report:
                return final_report, agent_log

    return final_report, agent_log


# ─── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Agent")
st.caption("Give a topic → agent searches the web → reads pages → writes a structured report.")

# Sample topics
with st.expander("💡 Sample topics"):
    samples = [
        "Impact of AI on software engineering jobs in India 2025",
        "Best practices for RAG systems in production",
        "Latest developments in agentic AI frameworks",
        "Python vs JavaScript for AI development",
        "State of GenAI adoption in Indian product companies"
    ]
    for s in samples:
        if st.button(s, key=s):
            st.session_state["topic"] = s

topic = st.text_input(
    "Research topic",
    value=st.session_state.get("topic", ""),
    placeholder="e.g. Impact of AI on software jobs in India"
)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("🚀 Research", type="primary", disabled=not topic.strip())

if run and topic.strip():
    st.divider()

    # Live progress
    st.subheader("⚙️ Agent Working...")
    progress = st.progress(0)
    log_box  = st.container()

    with st.spinner("Agent is researching..."):
        report, log = run_agent(topic.strip(), progress, log_box)

    progress.progress(1.0)

    if report:
        st.divider()
        st.subheader(f"📋 {report.get('title', 'Research Report')}")

        # Executive summary
        st.info(report.get("executive_summary", ""))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔑 Key Findings")
            for finding in report.get("key_findings", []):
                st.markdown(f"▸ {finding}")

        with col2:
            st.subheader("🔄 Perspectives")
            for p in report.get("perspectives", []):
                st.markdown(f"▸ {p}")

        st.subheader("📌 Conclusion")
        st.markdown(report.get("conclusion", ""))

        st.subheader("🔗 Sources")
        for url in report.get("sources", []):
            st.markdown(f"- [{url}]({url})")

        # Full agent log
        st.divider()
        with st.expander("🧠 Full agent reasoning log"):
            for entry in log:
                st.json(entry)

        # Download
        st.download_button(
            "💾 Download Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"research_{topic[:30].replace(' ','_')}.json",
            mime="application/json"
        )
    else:
        st.error("Agent did not produce a report. Try a different topic.")