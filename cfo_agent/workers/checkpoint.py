# workers/checkpoint.py
#
# This worker implements human-in-the-loop using a
# blocking input() call directly inside the node.
#
# HOW IT ACTUALLY WORKS:
# When the supervisor routes here, LangGraph runs this
# function like any other node. The input() call blocks
# the entire process — nothing moves forward until the
# human types yes or no in the terminal.
#
# This is the right approach for a CLI tool because:
# - Simple, no extra LangGraph configuration needed
# - The process stays alive while waiting for input
# - SQLite still checkpoints state before this node runs
#
# When would you use interrupt_before instead?
# In a web application where you cannot block a process.
# For example: a user submits a form hours later to approve.
# The process exits, state is saved to SQLite, and the
# graph resumes when the web endpoint is called again.
# For a terminal tool like this one, input() is cleaner.

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