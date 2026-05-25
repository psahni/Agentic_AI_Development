# core/agent.py
#
# PURPOSE:
# LangGraph agent that answers questions using only
# retrieved document chunks — never from LLM memory.
#
# THE GRAPH HAS THREE NODES:
#
# 1. retrieve_node
#    Takes the question, calls the retriever,
#    stores relevant chunks in state
#
# 2. answer_node
#    Takes the chunks, builds a grounded prompt,
#    calls Groq LLM, returns answer with citations
#
# 3. grade_node
#    Checks if the answer is actually grounded in
#    the retrieved chunks or if the LLM hallucinated
#    This is the anti-hallucination guard
#
# FLOW:
# Question → retrieve → answer → grade → final answer

import os
from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv

from pydantic import SecretStr
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from core.retriever import retrieve, format_context

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────
_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise EnvironmentError("GROQ_API_KEY is not set in environment")

llm = ChatGroq(
    model       = "llama-3.1-8b-instant",
    temperature = 0,       # zero temperature for factual answers
    api_key     = SecretStr(_api_key)
)


# ── State ─────────────────────────────────────────────────
# The backpack carried through every node.
# Simpler than CFO agent because the task is simpler —
# ask a question, get a grounded answer.

class RAGState(TypedDict):
    question:        str          # the user's question
    chunks:          list         # retrieved document chunks
    context:         str          # formatted chunks for LLM
    answer:          str          # LLM generated answer
    grounded:        bool         # did answer pass grounding check?
    collection_name: str          # which ChromaDB collection to search


# ── Node 1: Retrieve ──────────────────────────────────────

def retrieve_node(state: RAGState) -> dict:
    # Searches ChromaDB for chunks relevant to the question.
    # Stores both the raw chunks (for grading later) and
    # the formatted context string (for the LLM prompt).
    print("\n[Agent — Retrieve] Searching documents...")

    chunks  = retrieve(
        question        = state["question"],
        k               = 4,
        collection_name = state.get("collection_name", "documents")
    )
    context = format_context(chunks)

    return {
        "chunks":  chunks,
        "context": context
    }


# ── Node 2: Answer ────────────────────────────────────────

def answer_node(state: RAGState) -> dict:
    # Sends the retrieved context + question to the LLM.
    #
    # THE GROUNDING INSTRUCTION IS CRITICAL:
    # "Answer ONLY using the provided context"
    # "If the answer is not in the context, say so"
    # "Never use your own training knowledge"
    #
    # This is citation enforcement in action.
    # Without these instructions, the LLM will happily
    # blend retrieved content with its own memory —
    # making it impossible to trace where answers came from.

    print("\n[Agent — Answer] Generating grounded answer...")

    if not state["chunks"]:
        return {
            "answer":   "I could not find relevant information in the documents to answer this question.",
            "grounded": False
        }

    system_prompt = """You are a document analyst. Your job is to answer questions
strictly based on the provided document context.

STRICT RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say exactly:
   "This information is not available in the provided documents."
3. Never use your own training knowledge
4. Always reference which part of the context supports your answer
5. Be specific — include numbers, names, and dates from the context"""

    user_prompt = f"""DOCUMENT CONTEXT:
{state["context"]}

---

QUESTION: {state["question"]}

Answer based strictly on the context above.
At the end of your answer, add a SOURCES section listing
which chunks you used."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {"answer": str(response.content)}


# ── Node 3: Grade ─────────────────────────────────────────

def grade_node(state: RAGState) -> dict:
    # Checks whether the answer is grounded in retrieved chunks.
    #
    # WHY pass plain text instead of formatted context?
    # The formatted context has source labels and relevance
    # scores mixed in. Small LLMs get confused by metadata
    # and misjudge what is "in the context".
    # Plain text gives the grader a clean signal to work with.

    print("\n[Agent — Grade] Checking answer is grounded...")

    if not state["chunks"]:
        return {"grounded": False}

    # Extract plain text only — no source labels or scores
    plain_chunks = "\n\n".join([
        chunk["content"] for chunk in state["chunks"]
    ])

    grade_prompt = f"""Read the document excerpts and the answer below.

DOCUMENT EXCERPTS:
{plain_chunks}

ANSWER TO CHECK:
{state["answer"]}

Task: Does the answer contain ONLY facts that appear
in the document excerpts above?

Important: If the answer says something like
"I could not find this information" — reply YES
because that is a valid grounded response.

Reply with exactly one word: YES or NO"""

    response = llm.invoke([
        SystemMessage(content="You check if answers are supported by documents. Reply only YES or NO."),
        HumanMessage(content=grade_prompt)
    ])

    raw = str(response.content).strip().upper()
    print(f"   Grader raw response: '{raw}'")

    # If the answer itself says information is not available,
    # that IS a grounded response — the agent correctly
    # stayed within document bounds by admitting it does not know
    not_found_phrases = [
        "NOT AVAILABLE",
        "NOT FOUND",
        "CANNOT FIND",
        "NOT IN THE",
        "NO INFORMATION"
    ]

    answer_upper = state["answer"].upper()
    if any(phrase in answer_upper for phrase in not_found_phrases):
        print("   Answer correctly admitted information not found — marking grounded")
        return {"grounded": True}

    grounded = raw.startswith("YES")
    print(f"   Grounded: {grounded}")

    if not grounded:
        print("   ⚠️  Answer may contain information outside the documents")

    return {"grounded": grounded}

# ── Build Graph ───────────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("grade",    grade_node)

    # Simple sequential flow —
    # retrieve → answer → grade → done
    graph.add_edge(START,      "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer",   "grade")
    graph.add_edge("grade",    END)

    return graph.compile()


# ── Public Interface ──────────────────────────────────────
# This is what cli.py and api.py both call.
# Neither of them knows about LangGraph internals —
# they just call ask() and get an answer back.
# This is the decoupling pattern — one core, two doors.

_graph = None

def ask(question: str, collection_name: str = "documents") -> dict:
    global _graph
    if _graph is None:
        _graph = build_rag_graph()

    initial_state = RAGState(
        question        = question,
        chunks          = [],
        context         = "",
        answer          = "",
        grounded        = False,
        collection_name = collection_name
    )

    result = _graph.invoke(initial_state)

    return {
        "question": result["question"],
        "answer":   result["answer"],
        "grounded": result["grounded"],
        "sources":  len(result["chunks"])
    }