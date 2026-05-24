# main.py
#
# This file does three things:
# 1. Assembles the graph (nodes + edges + routing)
# 2. Attaches SQLite memory (checkpointer)
# 3. Handles human-in-the-loop via checkpoint_worker
#
# HOW SQLITE MEMORY WORKS HERE:
# SqliteSaver saves the entire state to memory.db
# after every node completes. If the program crashes
# or is interrupted, the next run with the same
# thread_id resumes from exactly where it stopped.
# Different thread_ids = completely separate sessions.

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import CFOState, get_initial_state
from supervisor import supervisor_node, route_to_worker
from workers.loader import loader_worker
from workers.analyst import analyst_worker
from workers.risk import risk_worker
from workers.checkpoint import checkpoint_worker
from workers.writer import writer_worker

PROJECT_ROOT = Path(__file__).resolve().parent


def build_graph(checkpointer):
    graph = StateGraph(CFOState)

    # ── Add Nodes ─────────────────────────────────────────────
    graph.add_node("supervisor",  supervisor_node)
    graph.add_node("loader",      loader_worker)
    graph.add_node("analyst",     analyst_worker)
    graph.add_node("risk",        risk_worker)
    graph.add_node("checkpoint",  checkpoint_worker)
    graph.add_node("writer",      writer_worker)

    # ── Entry Point ───────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Supervisor Routing ────────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_to_worker,
        {
            "loader":     "loader",
            "analyst":    "analyst",
            "risk":       "risk",
            "checkpoint": "checkpoint",
            "writer":     "writer",
            "FINISH":     END
        }
    )

    # ── Worker → Supervisor Loop ──────────────────────────────
    # Every worker returns control to the supervisor so it can
    # re-evaluate state and decide what runs next.
    graph.add_edge("loader",     "supervisor")
    graph.add_edge("analyst",    "supervisor")
    graph.add_edge("risk",       "supervisor")
    graph.add_edge("checkpoint", "supervisor")
    graph.add_edge("writer",     "supervisor")

    return graph.compile(checkpointer=checkpointer)


def main():
    db_path = str(PROJECT_ROOT / "memory.db")

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(checkpointer)

        thread_id     = "cfo-2024-annual"
        initial_state = get_initial_state(thread_id)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        print("Starting CFO Agent...\n")
        final_state = app.invoke(initial_state, config=config)

        if final_state.get("report"):
            print("\n=== CFO REPORT COMPLETE ===")
            print(final_state["report"])
        else:
            reason = final_state.get("error_message", "approval denied")
            print(f"\nReport not generated: {reason}")


if __name__ == "__main__":
    main()
