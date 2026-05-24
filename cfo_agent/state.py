# state.py
#
# The shared backpack carried through every node.
#
# THREE NEW FIELDS vs Sales Agent v2:
#
# thread_id   — identifies which session this run belongs to.
#               Like a conversation ID. SQLite uses this to
#               save and load the right checkpoint.
#               "cfo-jan-2025" and "cfo-feb-2025" are two
#               separate threads with separate memories.
#
# hitl_approved — tracks whether the human approved the
#               findings before the Writer runs.
#               Starts as False. Human sets it to True.
#               Writer checks this before proceeding.
#
# next_worker — same as Sales Agent v2. Supervisor writes
#               the name of the next worker here. LangGraph
#               reads it in the conditional edge to route.

from typing import TypedDict, Annotated
from operator import add


class CFOState(TypedDict):

    # ── Input ─────────────────────────────────────────────────
    thread_id:      str       # session identifier for SQLite

    # ── Data Pipeline ─────────────────────────────────────────
    raw_data:       str       # CSV loaded as JSON string
    metrics:        dict      # calculated financial ratios
    anomalies:      Annotated[list, add]  # accumulates findings
    insights:       str       # LLM narrative interpretation
    report:         str       # final written CFO report

    # ── Control ───────────────────────────────────────────────
    next_worker:    str       # supervisor's routing decision
    hitl_approved:  bool      # human approved findings?
    status:         str       # "ok" | "error"
    error_message:  str       # details if status is error


def get_initial_state(thread_id: str = "cfo-default") -> CFOState:
    return CFOState(
        thread_id     = thread_id,
        raw_data      = "",
        metrics       = {},
        anomalies     = [],
        insights      = "",
        report        = "",
        next_worker   = "loader",   # always start here
        hitl_approved = False,      # human has not approved yet
        status        = "ok",
        error_message = ""
    )