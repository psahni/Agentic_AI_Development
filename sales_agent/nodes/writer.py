# nodes/writer.py
# ============================================================
# NODE 5: REPORT WRITER
# Second LLM node. Takes all insights and anomalies and
# writes a complete, professional executive report.
# The output is clean markdown ready to share.
# ============================================================

import json
from datetime import datetime
from state import SalesState
from llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def writer_node(state: SalesState) -> dict:
    print("\n[Node 5 — Writer] Writing executive report with Groq LLM...")

    llm = get_llm()
    metrics = state["metrics"]

    system_prompt = """You are a CFO writing an executive sales 
performance report. Write in professional business language. 
Use markdown formatting with clear sections and headers. 
Be direct, data-driven, and action-oriented."""

    user_prompt = f"""
Write a complete executive sales performance report using this data.

METRICS: {json.dumps(metrics, indent=2)}

ANOMALIES: {chr(10).join(state['anomalies']) if state['anomalies'] else 'None detected'}

ANALYST INSIGHTS: {state['insights']}

Structure the report with these exact sections:
1. Executive Summary (3-4 sentences, headline numbers)
2. Revenue Performance (trends, target achievement)
3. Profitability Analysis (margins, expenses)
4. Key Anomalies & Risks (what needs attention)
5. Opportunities (what is working and should be amplified)
6. Recommendations (3-5 specific, actionable recommendations)

Report date: {datetime.now().strftime('%B %d, %Y')}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    print("   Report written successfully")
    return {**state, "report": response.content}
