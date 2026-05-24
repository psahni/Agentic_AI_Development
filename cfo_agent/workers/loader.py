# workers/loader.py
#
# WHY pandas?
# Pandas is the industry standard for tabular data in Python.
# It reads CSV into a DataFrame — a powerful structure that
# makes filtering, aggregation, and calculations simple.
#
# WHY store as JSON string in state?
# LangGraph state must be serializable — plain Python types
# only (str, dict, list, bool). A pandas DataFrame is not
# serializable. So we convert it to JSON string before
# storing, and convert back to DataFrame inside each worker.
#
# Think of it like packaging a product for shipping.
# The DataFrame is the product. JSON string is the box.
# Each worker unpacks the box, uses the product, repacks.

import io
import pandas as pd
from state import CFOState


def loader_worker(state: CFOState) -> dict:
    print("\n[Worker — Loader] Reading financial data...")

    try:
        df = pd.read_csv("data/financials.csv")

        # Validate required columns exist
        required = [
            "Month", "Revenue", "Net_Profit",
            "Gross_Margin", "Net_Margin",
            "Cash_Balance", "vs_Budget"
        ]
        missing = [c for c in required if c not in df.columns]

        if missing:
            return {
                "status":        "error",
                "error_message": f"Missing columns: {missing}"
            }

        total_revenue = df["Revenue"].sum()
        date_range    = f"{df['Month'].iloc[0]} → {df['Month'].iloc[-1]}"

        print(f"   Loaded {len(df)} months of data")
        print(f"   Date range:    {date_range}")
        print(f"   Total Revenue: ${total_revenue:,.0f}")

        return {
            "raw_data": df.to_json(),
            "status":   "ok"
        }

    except FileNotFoundError:
        return {
            "status":        "error",
            "error_message": "financials.csv not found in data/ folder"
        }