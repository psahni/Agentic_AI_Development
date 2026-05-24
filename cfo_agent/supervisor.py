# supervisor.py
#
# The Supervisor is the brain of the multi-agent system.
# It does NO actual work — no calculations, no writing.
#
# Its only job: look at current state and decide
# which worker runs next, or whether to stop.
#
# WHY USE AN LLM FOR ROUTING?
# Simple if/else routing breaks when requirements change.
# An LLM-based supervisor can handle nuance:
# "Skip risk assessment if no anomalies found"
# "Go straight to writer if human already approved"
# These decisions need reasoning, not just conditionals.
#
# HOW HITL FITS IN:
# After all workers complete, supervisor routes to
# "checkpoint" instead of "writer". This triggers
# the human review pause. Only after human approves
# does the supervisor route to "writer".

from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import supervisor_llm
from state import CFOState


def supervisor_node(state: CFOState) -> dict:
    print("\n[Supervisor] Evaluating state...")

    # Build a clear status summary for the LLM to reason about
    status_summary = f"""
    Pipeline status:
    - Data loaded:        {"YES" if state["raw_data"]      else "NO"}
    - Metrics calculated: {"YES" if state["metrics"]       else "NO"}
    - Anomalies found:    {len(state["anomalies"])} detected
    - Risk assessed:      {"YES" if state["metrics"] else "NO"}
    - Human approved:     {"YES" if state["hitl_approved"] else "NO"}
    - Report written:     {"YES" if state["report"]        else "NO"}
    - Current status:     {state["status"]}
    """

    system_prompt = """You are a supervisor managing a CFO financial analysis pipeline.

Your available workers are:
- loader     : reads and validates the financial CSV data
- analyst    : calculates all financial metrics and ratios
- risk       : detects anomalies, budget overruns, cash risks
- checkpoint : pauses for human review (use AFTER risk, BEFORE writer)
- writer     : generates insights and writes the final CFO report

Routing rules you must follow exactly:
1. loader must always run first
2. analyst runs after loader
3. risk runs after analyst
4. checkpoint runs after risk — always, no exceptions
5. writer runs only after human_approved is YES
6. Once report is written, respond FINISH
7. If status is error, respond FINISH immediately

Respond with ONLY one word — the worker name or FINISH.
No explanation. No punctuation. Just the single word."""

    human_prompt = f"""
Current pipeline state:
{status_summary}

Which worker should run next?
"""

    response = supervisor_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    next_worker = str(response.content).strip().lower()

    # Safety guard — if LLM returns unexpected value,
    # default to FINISH rather than getting stuck in a loop.
    # This is a critical production pattern — always have
    # a fallback for LLM unpredictability.
    valid = ["loader","analyst","risk","checkpoint","writer","finish"]
    if next_worker not in valid:
        print(f"   Unexpected response '{next_worker}' — defaulting to FINISH")
        next_worker = "finish"

    print(f"   Decision → {next_worker.upper()}")
    return {"next_worker": next_worker}


def route_to_worker(state: CFOState) -> str:
    # This function is called by LangGraph after every
    # supervisor run. It reads next_worker from state
    # and returns the node name to route to.
    # LangGraph uses this return value to follow
    # the correct conditional edge.
    decision = state["next_worker"].lower()
    if decision == "finish":
        return "FINISH"
    return decision

