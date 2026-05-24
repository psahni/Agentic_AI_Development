# graph.py
# ============================================================
# GRAPH WIRING
# This file only does one thing — connects all nodes
# into a LangGraph workflow. No business logic here.
# ============================================================

from langgraph.graph import StateGraph, START, END
from state import SalesState
from nodes.loader import loader_node
from nodes.calculator import calculator_node
from nodes.detector import detector_node
from nodes.analyst import analyst_node
from nodes.writer import writer_node
from nodes.saver import saver_node


def should_continue(state: SalesState) -> str:
    """Route to error handler if data load failed"""
    if state["status"] == "error":
        return "error"
    return "continue"


def error_node(state: SalesState) -> dict:
    print(f"\n[Error Handler] Pipeline stopped: {state['error_message']}")
    return dict(state)


def build_graph():
    graph = StateGraph(SalesState)

    # Register all nodes
    graph.add_node("loader",     loader_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("detector",   detector_node)
    graph.add_node("analyst",    analyst_node)
    graph.add_node("writer",     writer_node)
    graph.add_node("saver",      saver_node)
    graph.add_node("error",      error_node)

    # Entry point
    graph.add_edge(START, "loader")

    # After loader — check if data loaded OK
    graph.add_conditional_edges(
        "loader",
        should_continue,
        {
            "continue": "calculator",
            "error": "error"
        }
    )

    # Fixed pipeline after calculator
    graph.add_edge("calculator", "detector")
    graph.add_edge("detector",   "analyst")
    graph.add_edge("analyst",    "writer")
    graph.add_edge("writer",     "saver")
    graph.add_edge("saver",      END)
    graph.add_edge("error",      END)

    return graph.compile()