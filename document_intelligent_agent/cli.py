# cli.py
#
# PURPOSE:
# Interactive terminal interface for the Document
# Intelligence Agent.
#
# This file has zero business logic.
# It only handles input and output.
# The actual RAG work happens in core/agent.py.
#
# This separation means if you want to change how
# the agent works, you touch core/agent.py only.
# If you want to change how the terminal looks,
# you touch cli.py only. Clean boundaries.

# from core.agent import ask
from harness import harness_ask

def print_answer(result: dict):
    print("\n" + "─" * 50)
    print(f"Answer:   {result['answer']}")
    print(f"Grounded: {result['grounded']}")
    print(f"Sources:  {result['sources']} chunk(s) used")
    print(f"Duration: {result['duration_ms']}ms")
    print(f"Cost:     ${result['cost']['query_cost_usd']}")

    if result.get("warning"):
        print(f"Warning:  {result['warning']}")
    if result.get("error"):
        print(f"Error:    {result['error']}")
    print("─" * 50)


def run_cli():
    print("=" * 50)
    print("  DOCUMENT INTELLIGENCE AGENT")
    print("  Type your question. Type 'exit' to quit.")
    print("=" * 50)

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue
            if question.lower() in ["exit", "quit", "q"]:
                # Show cost summary before exiting
                from harness import get_session_summary
                summary = get_session_summary()
                print("\n── Session Summary ──────────────────────")
                print(f"  Queries:    {summary['total_queries']}")
                print(f"  Total cost: ${summary['total_cost_usd']}")
                print(f"  Avg cost:   ${summary['avg_cost_per_query']}")
                print("─────────────────────────────────────────")
                print("\nGoodbye.")
                break

            result = harness_ask(question)
            print_answer(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye.")
            break


if __name__ == "__main__":
    run_cli()


if __name__ == "__main__":
    run_cli()