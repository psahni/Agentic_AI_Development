from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
import json

load_dotenv()

client = Groq()

# ── Agent config ──────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert research assistant with access to web search.

Your job:
1. Break the topic into key questions to research
2. Search the web multiple times using different queries
3. Read the results carefully
4. Keep searching until you have enough to write a complete report
5. When done, write a structured report with sections and key findings

You have access to the tool: web_search(query)
Always search at least 3 times before writing the final report."""

# ── Tool definition (tells the model what tools exist) ────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a query. Returns titles, URLs and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ── Tool executor ─────────────────────────────────────────
def web_search(query: str) -> str:
    """Actually runs the web search and returns results as text."""
    print(f"  🔍 Searching: {query}")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n")
    return "\n---\n".join(results) if results else "No results found."

# ── Agentic loop ──────────────────────────────────────────
def run_research_agent(topic: str) -> str:
    """
    The core agentic loop:
    1. Send topic to Claude with tools available
    2. If Claude calls a tool → execute it → send results back
    3. Repeat until Claude stops calling tools
    4. Return Claude's final report
    """
    print(f"\n📋 Research topic: {topic}")
    print("=" * 50)

    # Start the conversation
    messages = [
        {"role": "user", "content": f"Research this topic thoroughly and write a report: {topic}"}
    ]

    # ── Loop ──────────────────────────────────────────────
    while True:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Available models:- https://console.groq.com/docs/models
            max_tokens=2048,
            tools=TOOLS,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Append Claude's response to conversation history
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": message.tool_calls if message.tool_calls else None
        })

        # ── Did Claude call a tool? ────────────────────────
        if finish_reason == "tool_calls" and message.tool_calls:
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                # Execute the tool
                if fn_name == "web_search":
                    result = web_search(fn_args["query"])
                else:
                    result = f"Unknown tool: {fn_name}"

                # Send tool result back into the conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        # ── Claude is done → return the final report ───────
        else:
            print("\n✅ Research complete!\n")
            return message.content


def format_report(topic: str, raw_report: str) -> str:
    """Ask the model to format the raw output into a clean structured report."""
    print("📝 Formatting report...")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=2048,
        messages=[
            {
                "role": "system",
                "content": "You are a professional report writer. Format research into clean markdown reports."
            },
            {
                "role": "user",
                "content": f"""Format this research into a clean markdown report with these exact sections:

# {topic}

## Executive Summary
(2-3 sentence overview)

## Key Findings
(bullet points of the most important facts)

## Detailed Analysis
(paragraphs expanding on the findings)

## Conclusion
(what this all means, looking ahead)

---
*Research conducted by AI Research Agent*

Here is the raw research to format:
{raw_report}"""
            }
        ]
    )
    return response.choices[0].message.content


def save_report(topic: str, report: str):
    """Save the report to a markdown file."""
    filename = topic.lower().replace(" ", "_")[:40] + "_report.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"💾 Report saved to: {filename}")
    return filename


# ── Run it ────────────────────────────────────────────────
if __name__ == "__main__":
    topic = "Impact of AI on software development in 2026"

    # Step 1: Run the agentic research loop
    raw_report = run_research_agent(topic)

    # Step 2: Format into clean structured report
    final_report = format_report(topic, raw_report)

    # Step 3: Print to terminal
    print("\n" + "=" * 50)
    print(final_report)

    # Step 4: Save to file
    save_report(topic, final_report)