# main.py
# ============================================================
# ENTRY POINT
# Run this file to execute the Sales Intelligence Agent.
# This file does nothing except kick off the graph.
# ============================================================

from graph import build_graph
from state import get_initial_state


def run_agent():
    print("=" * 50)
    print("  SALES INTELLIGENCE AGENT")
    print("=" * 50)

    graph = build_graph()
    result = graph.invoke(get_initial_state())

    if result["status"] == "error":
        print(f"\nAgent failed: {result['error_message']}")
    else:
        print("\n" + "=" * 50)
        print("  AGENT COMPLETED SUCCESSFULLY")
        print(f"  Report saved to: {result['report']}")
        print("=" * 50)


if __name__ == "__main__":
    run_agent()
    