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

from core.agent import ask


def print_answer(result: dict):
    # Formats and prints the agent result cleanly.
    print("\n" + "─" * 50)
    print(f"Answer:   {result['answer']}")
    print(f"Grounded: {result['grounded']}")
    print(f"Sources:  {result['sources']} chunk(s) used")
    print("─" * 50)


def run_cli():
    print("=" * 50)
    print("  DOCUMENT INTELLIGENCE AGENT")
    print("  Type your question. Type 'exit' to quit.")
    print("=" * 50)

    while True:
        try:
            # Get question from terminal
            question = input("\nYou: ").strip()

            # Exit conditions
            if not question:
                continue
            if question.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye.")
                break

            # Pass to agent — this is the only call this
            # file makes. Everything else is just formatting.
            result = ask(question)
            print_answer(result)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\nInterrupted. Goodbye.")
            break


if __name__ == "__main__":
    run_cli()