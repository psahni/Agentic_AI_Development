import pandas as pd
import os

data = {
    "Month": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ],
    "Revenue": [
        120000, 135000, 142000, 138000, 155000, 162000,
        98000,  165000, 170000, 145000, 188000, 210000
    ],
    "Target": [
        130000, 130000, 140000, 140000, 150000, 155000,
        160000, 165000, 170000, 175000, 180000, 200000
    ],
    "COGS": [
        48000, 52000, 55000, 54000, 60000, 63000,
        40000, 64000, 66000, 57000, 72000, 80000
    ],
    "Marketing_Expense": [
        12000, 13000, 14000, 13500, 15000, 15500,
        28000, 16000, 17000, 14500, 19000, 22000
    ],
    "Sales_Team_Cost": [
        25000, 25000, 25000, 25000, 26000, 26000,
        26000, 26000, 27000, 27000, 27000, 28000
    ],
    "Units_Sold": [
        400, 450, 470, 460, 515, 540,
        310, 550, 565, 480, 625, 700
    ]
}

df = pd.DataFrame(data)

# Calculate derived columns
df["Gross_Profit"] = df["Revenue"] - df["COGS"]
df["Total_Expenses"] = df["COGS"] + df["Marketing_Expense"] + df["Sales_Team_Cost"]
df["Net_Profit"] = df["Revenue"] - df["Total_Expenses"]
df["Profit_Margin_%"] = ((df["Net_Profit"] / df["Revenue"]) * 100).round(2)
df["vs_Target_%"] = (((df["Revenue"] - df["Target"]) / df["Target"]) * 100).round(2)

os.makedirs("data", exist_ok=True)
df.to_csv("data/sales_data.csv", index=False)
print("Sample data generated successfully!")
print(df.to_string())
