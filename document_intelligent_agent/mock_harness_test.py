import time
from unittest.mock import patch

# Simulate: first two calls fail, third succeeds
call_count = 0

# test_retry.py
import harness

call_count = 0

def mock_ask(question, collection_name="documents"):
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError(f"Simulated failure on attempt {call_count}")
    return {
        "question": question,
        "answer":   "This is the answer after retry.",
        "grounded": True,
        "sources":  1
    }

original_ask  = harness.ask
harness.ask   = mock_ask

try:
    result = harness.ask_with_retry("What is the revenue?")
    print(f"Success after {call_count} attempts")
    print(f"Answer: {result['answer']}")
except Exception as e:
    print(f"Failed: {e}")
finally:
    harness.ask = original_ask