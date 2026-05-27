# test_logger.py
from harness import log_interaction, get_recent_logs, LOG_FILE

# Log three fake interactions
interactions = [
    {
        "question":    "What is the total revenue?",
        "answer":      "The total revenue was 12.4 million dollars.",
        "grounded":    True,
        "sources":     2,
        "cost_data":   {"query_cost_usd": "0.00000688", "session_total_usd": "0.00000688"},
        "warning":     None,
        "error":       None,
        "duration_ms": 1240
    },
    {
        "question":    "What are the risk factors?",
        "answer":      "Three risk factors: competition, cloud costs, leadership risk.",
        "grounded":    True,
        "sources":     1,
        "cost_data":   {"query_cost_usd": "0.00000512", "session_total_usd": "0.00001200"},
        "warning":     None,
        "error":       None,
        "duration_ms": 980
    },
    {
        "question":    "Who is the CEO?",
        "answer":      "This information is not available in the provided documents.",
        "grounded":    True,
        "sources":     1,
        "cost_data":   {"query_cost_usd": "0.00000320", "session_total_usd": "0.00001520"},
        "warning":     None,
        "error":       None,
        "duration_ms": 870
    },
]

print("Logging 3 interactions...")
for i in interactions:
    log_interaction(**i)
print(f"Saved to: {LOG_FILE}\n")

print("Most recent 2 logs:")
recent = get_recent_logs(n=2)
for log in recent:
    print(f"  [{log['timestamp']}]")
    print(f"  Q: {log['question']}")
    print(f"  A: {log['answer'][:60]}")
    print(f"  Grounded: {log['grounded']} | Cost: {log['cost']['query_cost_usd']} | {log['duration_ms']}ms")
    print()

print(f"Total records in log file: {len(__import__('json').load(open(LOG_FILE)))}")
