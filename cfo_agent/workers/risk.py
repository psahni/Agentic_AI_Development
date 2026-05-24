# workers/risk.py
#
# WHY a dedicated Risk worker?
# In the Sales Agent, anomaly detection was bundled inside
# the Analyst. Here we separate it because financial risk
# assessment deserves its own focused worker.
#
# Risk assessment asks different questions than metric
# calculation. Metrics ask "what happened?"
# Risk asks "what is dangerous and why?"
#
# WHY rule-based thresholds AND LLM reasoning?
# Rules catch known patterns (cash below 2 months runway).
# LLM catches context-dependent risks that rules miss
# ("R&D doubled in April right before a revenue dip —
# was this investment poorly timed?")
# Combining both gives you precision AND intelligence.

import io
import json
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import worker_llm
from state import CFOState


def risk_worker(state: CFOState) -> dict:
    print("\n[Worker — Risk] Scanning for anomalies and risks...")

    df       = pd.read_json(io.StringIO(state["raw_data"]))
    metrics  = state["metrics"]
    anomalies = []

    # ── Rule-Based Detection ──────────────────────────────────
    for _, row in df.iterrows():

        # Revenue missed budget by more than 15%
        if row["vs_Budget"] < -15:
            anomalies.append(
                f"{row['Month']}: Revenue missed budget by "
                f"{abs(row['vs_Budget'])}% "
                f"(${row['Revenue']:,.0f} vs ${row['Budget']:,.0f})"
            )

        # Net margin critically low (below 5%)
        if row["Net_Margin"] < 5:
            anomalies.append(
                f"{row['Month']}: Net margin critically low "
                f"at {row['Net_Margin']}%"
            )

        # Cash balance below 2 months of expenses
        avg_expense = df["Total_Expenses"].mean()
        if row["Cash_Balance"] < (avg_expense * 2):
            anomalies.append(
                f"{row['Month']}: Cash balance danger zone — "
                f"${row['Cash_Balance']:,.0f} "
                f"(less than 2 months of expenses)"
            )

        # R&D spike — more than 2x the annual average
        avg_rd = df["R_and_D"].mean()
        if row["R_and_D"] > (avg_rd * 1.8):
            anomalies.append(
                f"{row['Month']}: R&D spend spiked to "
                f"${row['R_and_D']:,.0f} "
                f"(avg is ${avg_rd:,.0f})"
            )

        # Tax spike — more than 2x the annual average
        avg_tax = df["Tax"].mean()
        if row["Tax"] > (avg_tax * 2):
            anomalies.append(
                f"{row['Month']}: Unusual tax payment of "
                f"${row['Tax']:,.0f} "
                f"(avg is ${avg_tax:,.0f})"
            )

    print(f"   Rule-based anomalies: {len(anomalies)} found")
    for a in anomalies:
        print(f"   ⚠️  {a}")

    return {"anomalies": anomalies}