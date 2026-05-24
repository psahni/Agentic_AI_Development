# CFO Agent — Architecture Diagram

```
User Request
      ↓
┌─────────────────────────────┐
│         SUPERVISOR          │
│   Reads state + decides     │◄──────────────┐
└───┬──────────┬──────────────┘               │
    ↓          ↓          ↓                   │
┌────────┐ ┌────────┐ ┌─────────────┐         │
│ Loader │ │Analyst │ │   Risk      │         │
│        │ │        │ │  Assessor   │         │
│Reads + │ │Metrics │ │Checks cash  │         │
│validates│ │ratios  │ │runway +     │         │
│CSV     │ │burn    │ │critical     │         │
│        │ │rate    │ │thresholds   │         │
└────────┘ └────────┘ └─────────────┘         │
    │          │            │                  │
    └──────────┴────────────┘                  │
                ↓                              │
         Back to Supervisor ───────────────────┘
                ↓
    ┌───────────────────────┐
    │  ⏸ HUMAN CHECKPOINT   │  ← Agent pauses here
    │                       │    Shows findings
    │  "3 critical issues   │    Waits for approval
    │   found. Proceed?"    │    Human types YES/NO
    └───────────┬───────────┘
                ↓ (human approves)
         ┌─────────────┐
         │   Writer    │
         │             │
         │ Generates   │
         │ CFO report  │
         │ + saves     │
         └─────────────┘
                ↓
         ┌─────────────┐
         │   SQLite    │  ← State saved to disk
         │   Memory    │    Remembered forever
         └─────────────┘
```

## Component Responsibilities

| Component | Role |
|---|---|
| **Supervisor** | Orchestrates the graph; routes state between workers |
| **Loader** | Reads and validates `data/financials.csv` |
| **Analyst** | Computes metrics, ratios, and burn rate |
| **Risk Assessor** | Checks cash runway and flags critical thresholds |
| **Human Checkpoint** | Pauses execution; waits for YES/NO approval |
| **Writer** | Generates the CFO report and saves it to `reports/` |
| **SQLite Memory** | Persists full graph state across runs via LangGraph checkpointer |
