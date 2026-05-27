# api.py
#
# PURPOSE:
# FastAPI REST server that exposes the Document
# Intelligence Agent over HTTP.
#
# WHY FastAPI?
# - Fastest Python web framework for APIs
# - Auto-generates interactive documentation at /docs
# - Built-in request validation using Pydantic models
# - Industry standard for Python AI services
#
# TWO ENDPOINTS:
# GET  /health  → confirms server is running
# POST /ask     → accepts a question, returns answer
#
# This file has zero RAG logic.
# It only handles HTTP in and JSON out.
# All the work happens in core/agent.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from core.agent import ask
from harness import harness_ask

# ── App Setup ─────────────────────────────────────────────
# title and description appear in the auto-generated
# documentation at http://localhost:8000/docs
app = FastAPI(
    title       = "Document Intelligence Agent",
    description = "Ask questions about your documents. Answers grounded in retrieved content only.",
    version     = "1.0.0"
)


# ── Request and Response Models ───────────────────────────
# WHY define these as classes?
# FastAPI uses these to automatically validate incoming
# requests. If someone sends a request without a question
# field, FastAPI rejects it before it reaches our code.
# This is input validation — a key production pattern.

class QuestionRequest(BaseModel):
    question:        str
    collection_name: str = "documents"  # optional, defaults to documents


class AnswerResponse(BaseModel):
    question:    str
    answer:      str
    grounded:    bool
    sources:     int
    duration_ms: int
    cost:        dict
    warning:     str
    error:       str 

# ── Endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    # Simple health check endpoint.
    # WHY does this matter?
    # In production, monitoring systems ping /health
    # every 30 seconds to confirm the service is alive.
    # If it stops responding, alerts fire automatically.
    return {"status": "ok", "service": "Document Intelligence Agent"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    # Main endpoint — receives a question, returns an answer.
    #
    # WHY wrap in try/except?
    # If anything goes wrong inside the agent — retrieval
    # fails, LLM times out, ChromaDB is missing — we catch
    # it here and return a proper HTTP 500 error instead of
    # crashing the server. The server stays running for
    # other requests even when one request fails.
    # This is a critical production pattern.

    if not request.question.strip():
        raise HTTPException(
            status_code = 400,
            detail      = "Question cannot be empty"
        )

    try:
        result = harness_ask(
            question        = request.question,
            collection_name = request.collection_name
        )
        return AnswerResponse(
            question = result["question"],
            answer   = result["answer"],
            grounded = result["grounded"],
            sources  = result["sources"]
        )

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Agent error: {str(e)}"
        )