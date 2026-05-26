# harness.py
#
# Production harness for the Document Intelligence Agent.
#
# Wraps the agent with six protective layers:
# 1. Input Guard      — validates every question
# 2. Output Validator — checks every answer
# 3. Retry Logic      — handles LLM failures gracefully
# 4. Cost Tracker     — monitors token usage and spend
# 5. Structured Logger — auditable JSON log of every query
# 6. Harness Wrapper  — ties everything together
#
# cli.py and api.py call harness_ask() instead of ask().
# The agent itself is unchanged.

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from core.agent import ask

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_PATH    = PROJECT_ROOT / "logs"
LOGS_PATH.mkdir(exist_ok=True)

# ============================================================
# COMPONENT 1 — INPUT GUARD
# ============================================================

# Phrases that indicate a prompt injection attempt.
# These are attempts to override the agent's instructions.
# We block them before the agent ever sees the question.
INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore your instructions",
    "forget your rules",
    "you are now",
    "pretend you are",
    "act as if",
    "disregard your",
    "override your",
    "system prompt",
    "jailbreak",
]

MAX_QUESTION_LENGTH = 2000   # characters
MIN_QUESTION_LENGTH = 3      # characters


class InputGuardError(Exception):
    # Raised when a question fails validation.
    # Caught by the harness — never reaches the agent.
    pass


def validate_input(question: str) -> str:
    # Validates and sanitises the question.
    # Returns the cleaned question if valid.
    # Raises InputGuardError with a clear reason if not.
    #
    # WHY return the cleaned question rather than bool?
    # Because sanitisation (stripping whitespace, normalising
    # spacing) should happen here too. The agent receives
    # clean input regardless of what the user sent.

    # Strip leading and trailing whitespace
    cleaned = question.strip()

    # Check minimum length
    if len(cleaned) < MIN_QUESTION_LENGTH:
        raise InputGuardError(
            "Question is too short. Please ask a complete question."
        )

    # Check maximum length
    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise InputGuardError(
            f"Question is too long ({len(cleaned)} characters). "
            f"Please keep questions under {MAX_QUESTION_LENGTH} characters."
        )

    # Check for prompt injection attempts
    # WHY lowercase comparison?
    # "IGNORE PREVIOUS INSTRUCTIONS" is the same attack
    # as "ignore previous instructions". We normalise case
    # before checking so case variations do not slip through.
    lower = cleaned.lower()
    for phrase in INJECTION_PHRASES:
        if phrase in lower:
            raise InputGuardError(
                "Question contains restricted content and cannot be processed."
            )

    return cleaned