# test_cost.py
from harness import estimate_cost, get_session_summary

# Simulate three queries of different sizes
queries = [
    {
        "question": "What is the total revenue?",
        "context":  "ACME Corporation delivered strong results in 2024 " * 10,
        "answer":   "The total revenue was 12.4 million dollars.",
        "label":    "short query"
    },
    {
        "question": "What are all the risk factors and strategic priorities?",
        "context":  "Risk Factors: competition, cloud costs, key person risk. " * 20,
        "answer":   "There are three risk factors: competition, cloud costs, leadership risk. " * 5,
        "label":    "medium query"
    },
    {
        "question": "Summarise the entire document",
        "context":  "Full document content here " * 100,
        "answer":   "The document covers financials, risks, and strategy. " * 10,
        "label":    "large query"
    },
]

for q in queries:
    cost = estimate_cost(
        question      = q["question"],
        context       = q["context"],
        answer        = q["answer"],
        system_prompt = "You are a document analyst."
    )
    print(f"[{q['label']}]")
    print(f"  Input tokens:  {cost['input_tokens']}")
    print(f"  Output tokens: {cost['output_tokens']}")
    print(f"  Query cost:    ${cost['query_cost_usd']}")
    print(f"  Session total: ${cost['session_total_usd']}")
    print()

print("Session Summary:")
summary = get_session_summary()
for key, value in summary.items():
    print(f"  {key}: {value}")