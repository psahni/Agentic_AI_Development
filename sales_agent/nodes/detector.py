# nodes/detector.py
# ============================================================
# NODE 3: ANOMALY DETECTOR
# Finds unusual patterns purely through logic and thresholds.
# No LLM here — just rules applied to numbers.
# Anomalies get added to state["anomalies"] list.
# ============================================================

import pandas as pd
from state import SalesState
import io

def detector_node(state: SalesState) -> dict:
    print("\n[Node 3 — Detector] Scanning for anomalies...")

    df = pd.read_json(io.StringIO(state["raw_data"]))
    found = []

    # Rule 1 — Revenue more than 20% below target
    for _, row in df.iterrows():
        if row["vs_Target_%"] < -20:
            found.append(
                f"CRITICAL: {row['Month']} revenue missed target by "
                f"{abs(row['vs_Target_%'])}% "
                f"(${row['Revenue']:,.0f} vs ${row['Target']:,.0f} target)"
            )

    # Rule 2 — Month-on-month revenue drop greater than 15%
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        change = ((curr["Revenue"] - prev["Revenue"]) / prev["Revenue"]) * 100
        if change < -15:
            found.append(
                f"WARNING: {curr['Month']} revenue dropped "
                f"{abs(round(change, 1))}% from {prev['Month']} "
                f"(${prev['Revenue']:,.0f} → ${curr['Revenue']:,.0f})"
            )

    # Rule 3 — Marketing expense spike (double the average)
    avg_marketing = df["Marketing_Expense"].mean()
    for _, row in df.iterrows():
        if row["Marketing_Expense"] > avg_marketing * 1.8:
            found.append(
                f"WARNING: {row['Month']} marketing spend was "
                f"${row['Marketing_Expense']:,.0f} — "
                f"{round(row['Marketing_Expense']/avg_marketing, 1)}x above average"
            )

    # Rule 4 — Negative net profit month
    for _, row in df.iterrows():
        if row["Net_Profit"] < 0:
            found.append(
                f"CRITICAL: {row['Month']} had negative net profit: "
                f"${row['Net_Profit']:,.0f}"
            )

    # Rule 5 — Profit margin below 15%
    for _, row in df.iterrows():
        if 0 < row["Profit_Margin_%"] < 15:
            found.append(
                f"ALERT: {row['Month']} profit margin was only "
                f"{row['Profit_Margin_%']}% — below 15% threshold"
            )

    if found:
        print(f"   Found {len(found)} anomalies")
        for a in found:
            print(f"   → {a}")
    else:
        print("   No anomalies detected")

    return {**state, "anomalies": found}
