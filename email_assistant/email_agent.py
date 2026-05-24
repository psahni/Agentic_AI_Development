import os
import json
import time

from dotenv import load_dotenv
import google.generativeai as genai
from gmail_tools import search_emails, read_email, create_draft

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """You are a smart email assistant. Your job is to:
1. Search the inbox for recent emails
2. Read each email carefully
3. Categorize each email into one of these categories:
   - URGENT: Needs immediate attention or response
   - FOLLOW_UP: Needs a reply but not immediately
   - NEWSLETTER: Newsletters, promotions, subscriptions
   - NOTIFICATION: Automated system notifications
   - FYI: Informational, no action needed
4. For URGENT and FOLLOW_UP emails, draft a professional reply
5. Summarize all findings at the end

Always search first, then read individual emails, then categorize."""

# ── Tool definitions for Gemini ───────────────────────────
TOOLS = [
    {
        "name": "search_emails",
        "description": "Search Gmail inbox and return list of emails with subject, sender and preview.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Gmail search query e.g. 'in:inbox is:unread' or 'from:boss@company.com'"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of emails to return. Default 5."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_email",
        "description": "Read the full content of a specific email by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "email_id": {
                    "type": "string",
                    "description": "The Gmail message ID of the email to read"
                }
            },
            "required": ["email_id"]
        }
    },
    {
        "name": "create_draft",
        "description": "Save a draft reply in Gmail. Use for URGENT and FOLLOW_UP emails.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line, start with Re: for replies"
                },
                "body": {
                    "type": "string",
                    "description": "Full email body text"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
]


# ── Tool executor ─────────────────────────────────────────
def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Route tool calls to the correct Gmail function."""
    if tool_name == "search_emails":
        return search_emails(
            query=tool_args["query"],
            max_results=tool_args.get("max_results", 5)
        )
    elif tool_name == "read_email":
        return read_email(tool_args["email_id"])
    elif tool_name == "create_draft":
        return create_draft(
            to=tool_args["to"],
            subject=tool_args["subject"],
            body=tool_args["body"]
        )
    else:
        return f"Unknown tool: {tool_name}"

# ── Agentic loop ──────────────────────────────────────────
def run_email_agent():
    """
    Core agentic loop:
    1. Send task to Gemini with tools available
    2. If Gemini calls a tool → execute it → send results back
    3. Repeat until Gemini stops calling tools
    4. Return final categorized summary
    """
    print("\n📬 Starting Email Assistant Agent...")
    print("=" * 50)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        system_instruction=SYSTEM_PROMPT,
        tools=[{"function_declarations": TOOLS}]
    )

    # Start the conversation
    chat = model.start_chat()
    response = chat.send_message(
        "Check my inbox, read the latest 5 unread emails, "
        "categorize each one, and draft replies for any urgent "
        "or follow-up emails."
    )

    # ── Loop ──────────────────────────────────────────────
    while True:
        # Did Gemini call a tool?
        tool_called = False

        for part in response.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                tool_called = True
                fn_name = part.function_call.name
                fn_args = dict(part.function_call.args)

                print(f"\n🔧 Tool called: {fn_name}")

                # Execute the tool
                result = execute_tool(fn_name, fn_args)
                print(f"  ⏳ Waiting to avoid rate limit...")
                time.sleep(4)  # 4 seconds = safe under 15 RPM


                # Send result back to Gemini
                response = chat.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fn_name,
                                response={"result": result}
                            )
                        )]
                    )
                )
                break  # handle one tool at a time

        # No tool called → Gemini is done
        if not tool_called:
            print("\n✅ Agent finished!\n")
            return response.text



# ── Run it ────────────────────────────────────────────────
if __name__ == "__main__":
    summary = run_email_agent()
    print(summary)

    # Save the summary to a file
    with open("email_summary.txt", "w") as f:
        f.write(summary)
    print("\n💾 Summary saved to email_summary.txt")