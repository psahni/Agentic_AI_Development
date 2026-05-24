# nodes/analyst.py
# ============================================================
# NODE 4: ANALYST
# First LLM node. Groq reads the numbers and anomalies
# and generates real business insights — not just data,
# but what the data actually means for the business.
# ============================================================

import json
from state import SalesState
from llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def analyst_node(state: SalesState) -> dict:
    print("\n[Node 4 — Analyst] Generating insights with Groq LLM...")

    llm = get_llm()
    metrics = state["metrics"]
    anomalies = state["anomalies"]

    system_prompt = """You are a senior business analyst reviewing 
annual sales performance data. Your job is to interpret the numbers 
and explain what they mean for the business — not just repeat the 
numbers back. Focus on trends, risks, and opportunities. Be concise 
and direct. Write for a business audience, not a technical one."""

    user_prompt = f"""
Analyze this sales performance data and provide 5-7 key business insights.

ANNUAL METRICS:
- Total Revenue: ${metrics['total_revenue']:,}
- Total Net Profit: ${metrics['total_net_profit']:,}
- Average Profit Margin: {metrics['avg_profit_margin']}%
- Revenue Growth (Jan to Dec): {metrics['revenue_growth_%']}%
- Best Month: {metrics['best_month']} (${metrics['best_month_revenue']:,})
- Worst Month: {metrics['worst_month']} (${metrics['worst_month_revenue']:,})
- Months Above Target: {metrics['months_above_target']}/12
- Average vs Target: {metrics['avg_vs_target']}%

MONTHLY BREAKDOWN:
{json.dumps(metrics['monthly_summary'], indent=2)}

ANOMALIES DETECTED:
{chr(10).join(anomalies) if anomalies else 'None'}

Provide clear business insights. Each insight should explain WHY 
it matters, not just WHAT the number is.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    insights = response.content
    print("   Insights generated successfully")

    return {**state, "insights": insights}

