# nodes/calculator.py
# ============================================================
# NODE 2: CALCULATOR
# Pure pandas number crunching. No LLM here.
# Computes all key business metrics from raw data.
# Rule: calculate facts here, interpret them in analyst.py
# ============================================================

import json
import pandas as pd
from state import SalesState
import io

def calculator_node(state: SalesState) -> dict:
    print("\n[Node 2 — Calculator] Computing metrics...")

    df = pd.read_json(io.StringIO(state["raw_data"]))

    metrics = {
        # --- Annual Totals ---
        "total_revenue":        int(df["Revenue"].sum()),
        "total_expenses":       int(df["Total_Expenses"].sum()),
        "total_net_profit":     int(df["Net_Profit"].sum()),
        "total_units_sold":     int(df["Units_Sold"].sum()),

        # --- Averages ---
        "avg_monthly_revenue":  int(df["Revenue"].mean()),
        "avg_profit_margin":    round(df["Profit_Margin_%"].mean(), 2),

        # --- Best and Worst ---
        "best_month":           df.loc[df["Revenue"].idxmax(), "Month"],
        "best_month_revenue":   int(df["Revenue"].max()),
        "worst_month":          df.loc[df["Revenue"].idxmin(), "Month"],
        "worst_month_revenue":  int(df["Revenue"].min()),

        # --- Target Performance ---
        "months_above_target":  int((df["vs_Target_%"] > 0).sum()),
        "months_below_target":  int((df["vs_Target_%"] < 0).sum()),
        "avg_vs_target":        round(df["vs_Target_%"].mean(), 2),

        # --- Growth ---
        "revenue_growth_%":     round(
            ((df["Revenue"].iloc[-1] - df["Revenue"].iloc[0])
             / df["Revenue"].iloc[0]) * 100, 2
        ),

        # --- Expense Breakdown (annual) ---
        "total_cogs":           int(df["COGS"].sum()),
        "total_marketing":      int(df["Marketing_Expense"].sum()),
        "total_sales_team":     int(df["Sales_Team_Cost"].sum()),

        # --- Monthly detail for LLM context ---
        "monthly_summary": df[[
            "Month", "Revenue", "Target",
            "Net_Profit", "Profit_Margin_%", "vs_Target_%"
        ]].to_dict(orient="records")
    }

    print(f"   Annual Revenue:    ${metrics['total_revenue']:,.0f}")
    print(f"   Annual Profit:     ${metrics['total_net_profit']:,.0f}")
    print(f"   Avg Profit Margin: {metrics['avg_profit_margin']}%")
    print(f"   Best Month:        {metrics['best_month']}")
    print(f"   Worst Month:       {metrics['worst_month']}")

    return {**state, "metrics": metrics}

