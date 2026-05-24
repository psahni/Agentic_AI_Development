# nodes/loader.py
# ============================================================
# NODE 1: DATA LOADER
# Reads the CSV file and validates it has required columns.
# If something is wrong, sets status=error so the graph
# can route to the error handler instead of continuing.
# ============================================================

import pandas as pd
from state import SalesState


REQUIRED_COLUMNS = [
    "Month", "Revenue", "Target", "COGS",
    "Marketing_Expense", "Sales_Team_Cost",
    "Units_Sold", "Gross_Profit", "Total_Expenses",
    "Net_Profit", "Profit_Margin_%", "vs_Target_%"
]


def loader_node(state: SalesState) -> dict:
    print("\n[Node 1 — Loader] Reading sales data...")

    try:
        df = pd.read_csv("data/sales_data.csv")

        # Check all required columns exist
        missing = []
        for c in REQUIRED_COLUMNS:
            if c not in df.columns:
                missing.append(c)
        if missing:
            print(f"   ERROR: Missing columns: {missing}")
            return {
                **state,
                "status": "error",
                "error_message": f"Missing required columns: {missing}"
            }

        print(f"   Loaded {len(df)} months of data")
        print(f"   Date range: {df['Month'].iloc[0]} → {df['Month'].iloc[-1]}")
        print(f"   Total Revenue: ${df['Revenue'].sum():,.0f}")

        return {
            **state,
            "raw_data": df.to_json(),
            "status": "ok"
        }

    except FileNotFoundError:
        print("   ERROR: data/sales_data.csv not found")
        return {
            **state,
            "status": "error",
            "error_message": "File not found: data/sales_data.csv"
        }