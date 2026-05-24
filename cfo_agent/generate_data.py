# generate_data.py
#
# Generates 12 months of realistic financial data.
#
# WHY generate data programmatically?
# Because we know exactly what anomalies we planted.
# This lets us validate the agent is actually finding
# real issues — not just hallucinating insights.
#
# Anomalies planted:
# Apr → R&D budget doubles suddenly (58k vs normal 25k)
# Jul → Revenue crashes 35% but expenses stay high
# Oct → Cash balance drops dangerously (165k)
# Nov → Large tax payment causes near-loss month

import pandas as pd
import os

data = {
    "Month": [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ],
    "Revenue": [
        500000, 520000, 545000, 530000, 560000, 580000,
        375000, 590000, 610000, 625000, 640000, 700000
    ],
    "COGS": [
        175000, 182000, 190000, 185000, 196000, 203000,
        131000, 206000, 213000, 218000, 224000, 245000
    ],
    "Salaries": [
        120000, 120000, 120000, 122000, 122000, 122000,
        122000, 125000, 125000, 125000, 125000, 128000
    ],
    "Marketing": [
        40000, 42000, 44000, 43000, 45000, 46000,
        55000, 47000, 48000, 50000, 52000, 60000
    ],
    "Operations": [
        30000, 30000, 31000, 31000, 31000, 32000,
        32000, 32000, 33000, 33000, 33000, 35000
    ],
    "R_and_D": [
        25000, 25000, 25000, 58000, 26000, 26000,
        26000, 27000, 27000, 27000, 28000, 28000
    ],
    "Capex": [
        15000, 15000, 20000, 15000, 15000, 25000,
        15000, 15000, 20000, 15000, 15000, 30000
    ],
    "Tax": [
        18000, 19000, 20000, 19000, 20000, 21000,
        13000, 21000, 22000, 22000, 85000, 25000
    ],
    "Cash_Balance": [
        280000, 310000, 345000, 305000, 350000, 380000,
        220000, 390000, 420000, 165000, 118000, 280000
    ],
    "Budget": [
        480000, 495000, 510000, 525000, 540000, 555000,
        570000, 585000, 600000, 615000, 630000, 650000
    ]
}

df = pd.DataFrame(data)

# Derived columns — calculated automatically
df["Gross_Profit"]   = df["Revenue"] - df["COGS"]
df["Total_Expenses"] = (
    df["COGS"] + df["Salaries"] + df["Marketing"] +
    df["Operations"] + df["R_and_D"] + df["Capex"] + df["Tax"]
)
df["Net_Profit"]     = df["Revenue"] - df["Total_Expenses"]
df["Gross_Margin"]   = ((df["Gross_Profit"] / df["Revenue"]) * 100).round(2)
df["Net_Margin"]     = ((df["Net_Profit"]   / df["Revenue"]) * 100).round(2)
df["vs_Budget"]      = (((df["Revenue"] - df["Budget"]) / df["Budget"]) * 100).round(2)
df["Burn_Rate"]      = df["Total_Expenses"] - df["Revenue"]
df["Runway_Months"]  = (
    df["Cash_Balance"] / df["Total_Expenses"].mean()
).round(1)

os.makedirs("data", exist_ok=True)
df.to_csv("data/financials.csv", index=False)

print("=" * 55)
print("  Financial data generated successfully")
print("=" * 55)
print(f"\nColumns: {list(df.columns)}")
print(f"\nAnomalies planted:")
print("  Apr  — R&D doubled from $25k to $58k")
print("  Jul  — Revenue crashed 34% below budget target")
print("  Oct  — Cash balance dropped to $165k (danger zone)")
print("  Nov  — Tax spike of $85k causing near-loss month")
print(f"\nData preview:")
print(
    df[["Month","Revenue","Net_Profit","Net_Margin",
        "Cash_Balance","vs_Budget"]]
    .to_string(index=False)
)

