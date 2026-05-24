# utils/llm.py
#
# Shared LLM setup imported by all workers and supervisor.
#
# WHY TWO DIFFERENT MODELS?
#
# Supervisor uses llama-3.3-70b-versatile (larger model)
# because it makes the most critical decision in the
# system — routing. A wrong routing decision breaks the
# entire pipeline. Worth the extra cost and latency.
#
# Workers use llama-3.1-8b-instant (smaller model)
# because their tasks are focused and mechanical —
# calculate this, write that. The smaller model is
# 5x faster and cheaper. Perfect for structured tasks.
#
# This is called MODEL ROUTING — using the right model
# for the right job. A key cost optimisation pattern
# in production agent systems.

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Brain of the supervisor — smart, slower, more expensive
supervisor_llm = ChatGroq(
    model       = "llama-3.3-70b-versatile",
    temperature = 0,      # zero = fully deterministic routing
    api_key     = os.getenv("GROQ_API_KEY")
)

# Brain of workers — fast, cheap, focused
worker_llm = ChatGroq(
    model       = "llama-3.1-8b-instant",
    temperature = 0.1,    # slight variation for natural writing
    api_key     = os.getenv("GROQ_API_KEY")
)
