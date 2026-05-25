# workers/analyst.py
#
# WHY combine metrics AND burn rate in one worker?
# These calculations are tightly related — burn rate
# depends on the same expense columns as profit margin.
# Separating them would mean reading the DataFrame twice
# for no benefit. In multi-agent design, workers should
# be meaningful chunks, not micro-tasks.
#
# WHY no LLM here?
# Numbers do not need interpretation yet — just calculation.
# Pure pandas is faster, cheaper, and more accurate than
# asking an LLM to do arithmetic. LLMs are for reasoning
# about numbers, not computing them.

import io
import pandas as pd
from state import CFOState


def analyst_worker(state: CFOState) -> dict:
    print("\n[Worker — Analyst] Calculating financial metrics...")

    df = pd.read_json(io.StringIO(state["raw_data"]))

    # ── Core Metrics ──────────────────────────────────────────
    annual_revenue   = int(df["Revenue"].sum())
    annual_profit    = int(df["Net_Profit"].sum())
    avg_net_margin   = float(round(df["Net_Margin"].mean(), 2))
    avg_gross_margin = float(round(df["Gross_Margin"].mean(), 2))

    best_month  = str(df.loc[df["Net_Profit"].idxmax(), "Month"])
    worst_month = str(df.loc[df["Net_Profit"].idxmin(), "Month"])
    best_margin = str(df.loc[df["Net_Margin"].idxmax(), "Month"])

    # ── Revenue Growth ────────────────────────────────────────
    first_revenue  = df["Revenue"].iloc[0]
    last_revenue   = df["Revenue"].iloc[-1]
    revenue_growth = float(round(
        ((last_revenue - first_revenue) / first_revenue) * 100, 2
    ))

    # ── Burn Rate Analysis ────────────────────────────────────
    # Burn rate = months where expenses exceeded revenue
    # Positive burn rate = company spending more than earning
    burning_months = [str(m) for m in df[df["Burn_Rate"] > 0]["Month"].tolist()]

    # ── Cash Runway ───────────────────────────────────────────
    # How many months can the company survive at current
    # expense rate if revenue stopped tomorrow?
    avg_monthly_expense = df["Total_Expenses"].mean()
    current_cash        = df["Cash_Balance"].iloc[-1]
    cash_runway         = float(round(current_cash / avg_monthly_expense, 1))

    # ── Budget Performance ────────────────────────────────────
    months_above_budget = int((df["vs_Budget"] > 0).sum())
    months_below_budget = int((df["vs_Budget"] < 0).sum())
    avg_budget_variance = float(round(df["vs_Budget"].mean(), 2))

    # ── Monthly Breakdown (for report) ───────────────────────
    # Convert each row to native Python types — numpy scalars
    # are not msgpack serializable and will crash SQLite checkpointer.
    raw_monthly = df[[
        "Month", "Revenue", "Net_Profit",
        "Net_Margin", "Cash_Balance", "vs_Budget"
    ]].to_dict(orient="records")
    monthly = [
        {k: (str(v) if isinstance(v, str) else float(v) if hasattr(v, "item") else v)
         for k, v in row.items()}
        for row in raw_monthly
    ]

    metrics = {
        "annual_revenue":       annual_revenue,
        "annual_profit":        annual_profit,
        "avg_net_margin":       avg_net_margin,
        "avg_gross_margin":     avg_gross_margin,
        "best_month":           best_month,
        "worst_month":          worst_month,
        "best_margin_month":    best_margin,
        "revenue_growth_pct":   revenue_growth,
        "burning_months":       burning_months,
        "cash_runway_months":   cash_runway,
        "months_above_budget":  months_above_budget,
        "months_below_budget":  months_below_budget,
        "avg_budget_variance":  avg_budget_variance,
        "monthly_breakdown":    monthly
    }

    print(f"   Annual Revenue:    ${annual_revenue:,.0f}")
    print(f"   Annual Profit:     ${annual_profit:,.0f}")
    print(f"   Avg Net Margin:    {avg_net_margin}%")
    print(f"   Revenue Growth:    {revenue_growth}%")
    print(f"   Cash Runway:       {cash_runway} months")
    print(f"   Budget Variance:   {avg_budget_variance}%")

    return {"metrics": metrics}



