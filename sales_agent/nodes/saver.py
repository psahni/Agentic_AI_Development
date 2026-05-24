# nodes/saver.py
# ============================================================
# NODE 6: SAVER
# Saves the final report to disk.
# Simple, focused, no LLM needed here.
# ============================================================

import os
from datetime import datetime
from state import SalesState


def saver_node(state: SalesState) -> dict:
    print("\n[Node 6 — Saver] Saving report to disk...")

    os.makedirs("reports", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/sales_report_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(f"# Sales Intelligence Report\n")
        f.write(f"*Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}*\n\n")
        f.write(state["report"])

    print(f"   Report saved: {filename}")
    return {**state, "report": filename}
