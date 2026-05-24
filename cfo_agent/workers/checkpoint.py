# workers/checkpoint.py
#
# This is where the agent PAUSES and asks a human
# to review findings before the report is written.
#
# WHY human-in-the-loop here specifically?
# The Writer uses the anomalies and metrics to generate
# a boardroom-ready CFO report. If the anomaly list
# contains incorrect findings, the report will too.
# A human review at this point catches errors BEFORE
# they end up in an executive document.
#
# HOW IT WORKS IN LANGGRAPH:
# In main.py, when we compile the graph, we pass
# interrupt_before=["checkpoint"] to the checkpointer.
# This tells LangGraph: before running this node,
# PAUSE the entire graph and save state to SQLite.
# The program exits. When you run it again with the
# same thread_id, LangGraph resumes from this exact
# point — with the human's input included.
#
# This is different from a simple input() call because:
# - State is fully saved to disk during the pause
# - The agent can be paused for hours or days
# - Multiple humans can review on different machines
# - The graph resumes exactly where it stopped

from state import CFOState


def checkpoint_worker(state: CFOState) -> dict:
    print("\n" + "=" * 55)
    print("  ⏸  HUMAN CHECKPOINT")
    print("=" * 55)

    metrics   = state["metrics"]
    anomalies = state["anomalies"]

    print(f"\n  Financial Summary:")
    print(f"  Annual Revenue : ${metrics.get('annual_revenue', 0):,.0f}")
    print(f"  Annual Profit  : ${metrics.get('annual_profit',  0):,.0f}")
    print(f"  Avg Net Margin : {metrics.get('avg_net_margin',  0)}%")
    print(f"  Cash Runway    : {metrics.get('cash_runway_months', 0)} months")

    print(f"\n  Anomalies Detected ({len(anomalies)}):")
    if anomalies:
        for i, a in enumerate(anomalies, 1):
            print(f"  {i}. {a}")
    else:
        print("  None detected")

    print("\n" + "-" * 55)

    # This is where human provides input
    # The graph has paused — human reads the findings
    # and decides whether to proceed with the report
    while True:
        response = input(
            "\n  Approve these findings and generate report? "
            "(yes / no): "
        ).strip().lower()

        if response in ["yes", "y"]:
            print("  ✅ Approved — proceeding to report generation")
            return {"hitl_approved": True}

        elif response in ["no", "n"]:
            print("  ❌ Rejected — stopping pipeline")
            return {
                "hitl_approved": False,
                "status":        "error",
                "error_message": "Human rejected findings at checkpoint"
            }

        else:
            print("  Please type 'yes' or 'no'")