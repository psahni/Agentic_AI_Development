# workers/writer.py
#
# WHY use LLM only in the Writer, not in earlier workers?
# Earlier workers dealt with facts — load CSV, calculate
# numbers, detect threshold breaches. Facts do not need
# an LLM. The Writer deals with MEANING — what should a
# CFO understand from these numbers? What actions should
# leadership take? That requires language intelligence.
#
# TWO LLM CALLS IN ONE WORKER:
# Call 1 → Generate insights (what the numbers MEAN)
# Call 2 → Write the full report (structured document)
# We separate these because insights feed into the report.
# If we combined them, the LLM would try to analyse AND
# write simultaneously — lower quality on both tasks.
# Two focused calls beats one unfocused call every time.

import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import worker_llm
from state import CFOState


def writer_worker(state: CFOState) -> dict:
    print("\n[Worker — Writer] Generating insights and CFO report...")

    metrics   = state["metrics"]
    anomalies = state["anomalies"]

    # ── LLM Call 1: Generate Insights ────────────────────────
    # We give the LLM numbers and anomalies.
    # We ask for INTERPRETATION — what do these mean
    # for the business? What patterns matter?
    # This is where the agent earns its value —
    # turning raw numbers into business intelligence.

    insights_prompt = f"""
You are a senior CFO analyst. Review these financial metrics
and anomalies, then write 4 concise business insights.

ANNUAL METRICS:
- Revenue:         ${metrics.get('annual_revenue', 0):,.0f}
- Net Profit:      ${metrics.get('annual_profit',  0):,.0f}
- Avg Net Margin:  {metrics.get('avg_net_margin',  0)}%
- Revenue Growth:  {metrics.get('revenue_growth_pct', 0)}%
- Cash Runway:     {metrics.get('cash_runway_months', 0)} months
- Budget Variance: {metrics.get('avg_budget_variance', 0)}%
- Best Month:      {metrics.get('best_month', '')}
- Worst Month:     {metrics.get('worst_month', '')}

ANOMALIES DETECTED:
{chr(10).join(f"- {a}" for a in anomalies) if anomalies else "None"}

Rules:
- Write in plain business English
- Each insight is 2-3 sentences maximum
- Focus on trends, risks, and opportunities
- Be specific — reference actual numbers
- No bullet points — write as short paragraphs
"""

    insights_response = worker_llm.invoke([
        SystemMessage(content="You are a senior CFO analyst."),
        HumanMessage(content=insights_prompt)
    ])
    insights = insights_response.content
    print("   Insights generated")

    # ── LLM Call 2: Write the Full Report ─────────────────────
    # Now we give the LLM everything — metrics, anomalies,
    # and the insights just generated — and ask it to produce
    # a structured, boardroom-ready markdown report.
    #
    # WHY provide the insights from Call 1?
    # So the LLM builds on the analysis rather than
    # redoing it. Each call builds on the previous one.
    # This is called CHAINING — a core prompting pattern.

    monthly_table = "\n".join([
        f"| {m['Month']} | ${m['Revenue']:,.0f} | "
        f"${m['Net_Profit']:,.0f} | {m['Net_Margin']:.1f}% | "
        f"${m['Cash_Balance']:,.0f} | {m['vs_Budget']:.1f}% |"
        for m in metrics.get("monthly_breakdown", [])
    ])

    report_prompt = f"""
Write a professional CFO financial report in markdown format.
Use this exact structure and be specific with numbers.

# CFO Financial Performance Report
## Executive Summary
(3-4 sentences covering the year at a glance)

## Key Annual Metrics
(present the core numbers clearly)

## Monthly Performance
| Month | Revenue | Net Profit | Net Margin | Cash Balance | vs Budget |
|-------|---------|------------|------------|--------------|-----------|
{monthly_table}

## Anomalies & Risk Flags
(list each anomaly with business impact)

## Analyst Insights
{insights}

## Strategic Recommendations
(3-4 specific, actionable recommendations for leadership)

---
DATA PROVIDED:
Metrics: {json.dumps({k: v for k, v in metrics.items() if k != 'monthly_breakdown'}, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))}
Anomalies: {json.dumps(anomalies, indent=2)}
"""

    report_response = worker_llm.invoke([
        SystemMessage(content="You are a professional CFO report writer."),
        HumanMessage(content=report_prompt)
    ])
    report = str(report_response.content)
    print("   Report written")

    # ── Save Report to Disk ───────────────────────────────────
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = reports_dir / f"cfo_report_{timestamp}.md"

    filepath.write_text(report, encoding="utf-8")

    print(f"   Report saved → {filepath}")

    return {
        "insights": insights,
        "report":   report
    }

