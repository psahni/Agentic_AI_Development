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


# ============================================================
# COMPONENT 2 — OUTPUT VALIDATOR
# ============================================================

MIN_ANSWER_LENGTH = 10    # characters

# Phrases that indicate the agent correctly admitted
# it could not find information. These are VALID responses
# even though they are short — do not flag them.
NOT_FOUND_PHRASES = [
    "not available in the provided",
    "not found in the",
    "cannot find",
    "no information",
    "not in the documents",
]


class OutputValidationError(Exception):
    # Raised when an answer fails validation.
    # The harness catches this and triggers a retry
    # or returns a safe fallback message.
    pass


def validate_output(answer: str, grounded: bool) -> dict:
    # Validates the agent's answer before returning to user.
    # Returns a dict with the answer and a warning flag.
    #
    # WHY return a dict instead of raising an exception?
    # Unlike input failures (which should always be blocked),
    # output issues are nuanced. A short answer might be
    # correct ("Yes." is a valid answer to a yes/no question).
    # An ungrounded answer might still be useful with a warning.
    # We return the answer WITH a warning flag rather than
    # blocking it entirely — the caller decides what to do.

    result = {
        "answer":   answer,
        "warning":  None,
        "valid":    True
    }

    # Check for empty answer
    if not answer or not answer.strip():
        result["valid"]   = False
        result["warning"] = "Agent returned an empty response."
        result["answer"]  = (
            "I was unable to generate a response. "
            "Please try rephrasing your question."
        )
        return result

    # Check for suspiciously short answer
    # (but allow known not-found phrases which can be short)
    lower = answer.lower()
    is_not_found = any(phrase in lower for phrase in NOT_FOUND_PHRASES)

    if len(answer.strip()) < MIN_ANSWER_LENGTH and not is_not_found:
        result["warning"] = (
            f"Answer is unusually short ({len(answer.strip())} characters). "
            "Response may be incomplete."
        )

    # Check grounding signal from grade node
    # If graded as not grounded AND it is not a valid
    # not-found response, attach a reliability warning.
    if not grounded and not is_not_found:
        result["warning"] = (
            "This answer may contain information not found "
            "in the provided documents. Verify before relyi ng on it."
        )

    return result



# ============================================================
# COMPONENT 3 — RETRY LOGIC
# ============================================================

MAX_RETRIES    = 3      # maximum number of attempts
BASE_DELAY     = 1.0    # seconds to wait after first failure
BACKOFF_FACTOR = 2.0    # multiply delay by this after each failure

# Delay sequence: 1s → 2s → 4s
# After 3 failures we give up and return an error


def ask_with_retry(question: str,
                   collection_name: str = "documents") -> dict:
    # Calls the agent with automatic retry on failure.
    #
    # WHY separate this from the main harness function?
    # Retry logic is a distinct concern from validation and
    # logging. Keeping it in its own function makes it easy
    # to adjust retry behaviour without touching anything else.
    # It also makes it testable in isolation.

    last_error  = None
    delay       = BASE_DELAY

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            if attempt > 1:
                print(f"   Retry attempt {attempt}/{MAX_RETRIES} "
                      f"(waiting {delay:.1f}s...)")
                time.sleep(delay)
                delay *= BACKOFF_FACTOR

            result = ask(
                question        = question,
                collection_name = collection_name
            )
            return result

        except InputGuardError:
            # Never retry input validation failures —
            # the question itself is the problem
            raise

        except Exception as e:
            last_error = e
            print(f"   Attempt {attempt} failed: {type(e).__name__}: {e}")

            if attempt == MAX_RETRIES:
                # All retries exhausted — give up
                print(f"   All {MAX_RETRIES} attempts failed.")
                raise RuntimeError(
                    f"Agent failed after {MAX_RETRIES} attempts. "
                    f"Last error: {last_error}"
                ) from last_error

    # Unreachable — the loop always returns or raises on the final attempt.
    # Required so the type checker sees a guaranteed exit on every code path.
    raise RuntimeError("ask_with_retry exited loop without returning")


# ============================================================
# COMPONENT 4 — COST TRACKER
# ============================================================

# Groq pricing for llama-3.1-8b-instant (per million tokens)
# Update these if pricing changes or if you switch models.
# Source: groq.com/pricing
COST_PER_MILLION_INPUT_TOKENS  = 0.05   # USD
COST_PER_MILLION_OUTPUT_TOKENS = 0.08   # USD

# Running totals — accumulate across the session
_session_total_cost        = 0.0
_session_total_queries     = 0
_session_total_input_tokens  = 0
_session_total_output_tokens = 0


def estimate_cost(question:       str,
                  context:        str,
                  answer:         str,
                  system_prompt:  str = "") -> dict:
    # Estimates token usage and cost for one query.
    #
    # WHY estimate rather than use exact token counts?
    # Groq does not always return token usage in the response.
    # The 1 token ≈ 4 characters rule is a reliable estimate
    # used industry-wide for cost forecasting. For exact
    # billing, check your Groq dashboard.
    #
    # Input tokens = everything sent TO the LLM:
    #   system prompt + context + question
    # Output tokens = everything received FROM the LLM:
    #   the answer

    global _session_total_cost
    global _session_total_queries
    global _session_total_input_tokens
    global _session_total_output_tokens

    # Estimate token counts
    input_text    = system_prompt + context + question
    input_tokens  = len(input_text) // 4
    output_tokens = len(answer) // 4

    # Calculate cost in USD
    input_cost  = (input_tokens  / 1_000_000) * COST_PER_MILLION_INPUT_TOKENS
    output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT_TOKENS
    query_cost  = input_cost + output_cost

    # Update session totals
    _session_total_cost          += query_cost
    _session_total_queries       += 1
    _session_total_input_tokens  += input_tokens
    _session_total_output_tokens += output_tokens

    cost_data = {
        "input_tokens":     input_tokens,
        "output_tokens":    output_tokens,
        "query_cost_usd":   f"{query_cost:.8f}", #round(query_cost, 6),
        "session_total_usd": f"{_session_total_cost:.8f}", # round(_session_total_cost, 6),
        "session_queries":  _session_total_queries
    }

    # Warn if a single query is unusually expensive
    # This catches runaway context or extremely long answers
    if query_cost > 0.01:
        print(f"   ⚠️  High cost query: ${query_cost:.6f}")

    return cost_data


def get_session_summary() -> dict:
    # Returns the full cost summary for the current session.
    # Useful for printing at the end of a CLI session
    # or exposing on a /stats API endpoint.
    return {
        "total_queries":       _session_total_queries,
        "total_input_tokens":  _session_total_input_tokens,
        "total_output_tokens": _session_total_output_tokens,
        "total_cost_usd":      f"{_session_total_cost:.8f}", #round(_session_total_cost, 6),
        "avg_cost_per_query":  f"{_session_total_cost / _session_total_queries:.8f}" if _session_total_queries > 0 else 0.0
    }


# ============================================================
# COMPONENT 5 — STRUCTURED LOGGER
# ============================================================

LOG_FILE = LOGS_PATH / "agent_interactions.json"


def log_interaction(question:    str,
                    answer:      str,
                    grounded:    bool,
                    sources:     int,
                    cost_data:   dict,
                    warning:     str  = "",
                    error:       str  = "",
                    duration_ms: int  = 0) -> None:
    # Appends one interaction record to the JSON log file.
    #
    # WHY append rather than overwrite?
    # Every run adds to the same log file. This gives you
    # a complete history across all sessions — not just
    # the current one.
    #
    # WHY store duration_ms?
    # Latency is a key production metric. If queries suddenly
    # take 10 seconds instead of 2, you want to know when
    # that started happening and for which question types.

    record = {
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "question":     question,
        "answer":       answer,
        "grounded":     grounded,
        "sources_used": sources,
        "duration_ms":  duration_ms,
        "cost":         cost_data,
        "warning":      warning,
        "error":        error
    }

    # Read existing log, append new record, write back.
    # WHY read-then-write instead of append mode?
    # Append mode writes raw JSON objects separated by
    # newlines — valid NDJSON but harder to parse.
    # Read-then-write keeps the file as a proper JSON array
    # that any JSON parser can load in one line.
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    existing.append(record)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def get_recent_logs(n: int = 5) -> list:
    # Returns the n most recent log entries.
    # Useful for a /logs endpoint in the API
    # or a --history flag in the CLI.
    if not LOG_FILE.exists():
        return []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        try:
            logs = json.load(f)
            return logs[-n:]
        except json.JSONDecodeError:
            return []

# ============================================================
# COMPONENT 6 — HARNESS WRAPPER
# ============================================================

def harness_ask(question:        str,
                collection_name: str = "documents") -> dict:
    # The single entry point for the entire production agent.
    #
    # Orchestrates all five components in order:
    # 1. Input Guard      — validate the question
    # 2. Retry Logic      — call agent with automatic retry
    # 3. Output Validator — check the answer quality
    # 4. Cost Tracker     — estimate tokens and cost
    # 5. Structured Logger — write everything to disk
    #
    # Returns a clean dict the caller can use directly.
    # Errors are caught here — callers never see raw exceptions.

    start_time = time.time()
    warning    = None
    error      = None

    # ── Step 1: Input Guard ───────────────────────────────────
    try:
        clean_question = validate_input(question)
    except InputGuardError as e:
        # Input failed validation — log and return immediately.
        # No point calling the agent with a bad question.
        duration_ms = int((time.time() - start_time) * 1000)
        log_interaction(
            question    = question,
            answer      = "",
            grounded    = False,
            sources     = 0,
            cost_data   = {"query_cost_usd": "0.00000000",
                           "session_total_usd": "0.00000000"},
            warning     = "",
            error       = str(e),
            duration_ms = duration_ms
        )
        return {
            "question": question,
            "answer":   str(e),
            "grounded": False,
            "sources":  0,
            "warning":  None,
            "error":    str(e),
            "cost":     {"query_cost_usd": "0.00000000"},
            "duration_ms": duration_ms
        }

    # ── Step 2: Call Agent with Retry ─────────────────────────
    try:
        result = ask_with_retry(
            question        = clean_question,
            collection_name = collection_name
        )
    except RuntimeError as e:
        # All retries exhausted — log the failure and return
        # a safe fallback message instead of crashing.
        duration_ms = int((time.time() - start_time) * 1000)
        error_msg   = "The agent is temporarily unavailable. Please try again shortly."
        log_interaction(
            question    = clean_question,
            answer      = "",
            grounded    = False,
            sources     = 0,
            cost_data   = {"query_cost_usd": "0.00000000",
                           "session_total_usd": "0.00000000"},
            warning     = "",
            error       = str(e),
            duration_ms = duration_ms
        )
        return {
            "question":    clean_question,
            "answer":      error_msg,
            "grounded":    False,
            "sources":     0,
            "warning":     None,
            "error":       str(e),
            "cost":        {"query_cost_usd": "0.00000000"},
            "duration_ms": duration_ms
        }

    # ── Step 3: Output Validator ──────────────────────────────
    validated = validate_output(
        answer   = result["answer"],
        grounded = result["grounded"]
    )
    warning = validated["warning"]

    # ── Step 4: Cost Tracker ──────────────────────────────────
    # Use engineered_context if available, fall back to
    # raw context. engineered_context is richer and gives
    # a more accurate token estimate.
    context_for_cost = result.get("engineered_context", "") or \
                       result.get("context", "")

    cost_data = estimate_cost(
        question      = clean_question,
        context       = context_for_cost,
        answer        = validated["answer"],
        system_prompt = "You are a document analyst."
    )

    # ── Step 5: Structured Logger ─────────────────────────────
    duration_ms = int((time.time() - start_time) * 1000)

    log_interaction(
        question    = clean_question,
        answer      = validated["answer"],
        grounded    = result["grounded"],
        sources     = result["sources"],
        cost_data   = cost_data,
        warning     = warning,
        error       = "",
        duration_ms = duration_ms
    )

    # ── Return clean result ───────────────────────────────────
    return {
        "question":    clean_question,
        "answer":      validated["answer"],
        "grounded":    result["grounded"],
        "sources":     result["sources"],
        "warning":     warning,
        "cost":        cost_data,
        "duration_ms": duration_ms,
        "error":       None
    }
