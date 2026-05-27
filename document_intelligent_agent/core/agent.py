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
    engineered_context: str
    answer:          str          # LLM generated answer
    grounded:        bool         # did answer pass grounding check?
    collection_name: str          # which ChromaDB collection to search

#─────────────────────────────────────────────────────────────

def expand_query(question: str) -> str:
    # Rewrites short or vague queries into more specific ones
    # before sending to the retriever.
    #
    # WHY query expansion?
    # Short queries like "total revenue?" match many chunks
    # because the keywords appear in multiple contexts.
    # Expanding the query adds specificity that guides the
    # retriever toward the most relevant chunk.
    #
    # Example:
    # "What is the total revenue?" (vague)
    # → "What is the total annual revenue figure in dollars?" (specific)

    if len(question.split()) > 8:
        # Long questions are already specific enough
        return question

    expansion_prompt = f"""Rewrite this question to be more specific 
and descriptive for document search. Add context about what kind 
of answer is expected. Keep it as one sentence under 20 words.

Original: {question}
Rewritten:"""

    response = llm.invoke([
        SystemMessage(content="You rewrite vague questions into specific search queries. One sentence only."),
        HumanMessage(content=expansion_prompt)
    ])

    expanded = str(response.content).strip()
    print(f"   Query expanded: '{question}' → '{expanded}'")
    return expanded


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


# ───────────────────────────────────────────────────

def context_engineer_node(state: RAGState) -> dict:
    # PURPOSE:
    # Transforms raw retrieved chunks into an optimised
    # context before the LLM sees it.
    #
    # This is where we make deliberate decisions about
    # what goes into the context window, in what order,
    # and how it is formatted.
    #
    # Raw retrieval = giving the LLM a pile of papers
    # Context engineering = giving the LLM a prepared brief

    print("\n[Context Engineer] Optimising context...")

    chunks = state["chunks"]

    if not chunks:
        return {"engineered_context": "No relevant content found."}

    # ── Job 1: Rank by relevance score ───────────────────────
    # Sort descending — highest relevance first.
    # LLMs attend more strongly to early context.
    # Most important evidence goes at the top.
    ranked = sorted(chunks, key=lambda x: x["score"], reverse=True)

    print(f"   Ranked {len(ranked)} chunks by relevance")
    for i, c in enumerate(ranked, 1):
        print(f"   {i}. score {c['score']:.3f} — "
              f"{c['content'][:60].replace(chr(10),' ')}...")

    # ── Job 2: Compress low-relevance chunks ─────────────────
    # Chunks above 0.5 → keep full text (high signal)
    # Chunks below 0.5 → summarise to one sentence (low signal)
    #
    # WHY 0.5 as the threshold?
    # In practice, scores above 0.5 consistently contain
    # direct answers. Scores below 0.5 are contextually
    # related but often not directly useful. Compressing
    # them preserves the signal without the token cost.

    HIGH_THRESHOLD = 0.5 # If it does not return right ans you can check, and try to lower the value, it will bring more chunks into context
    context_sections = []

    for i, chunk in enumerate(ranked, 1):
        if chunk["score"] >= HIGH_THRESHOLD:
            # Full text — high confidence it is relevant
            section = (
                f"[Passage {i} — relevance {chunk['score']:.2f}]\n"
                f"{chunk['content']}"
            )
            print(f"   Passage {i}: kept full ({len(chunk['content'])} chars)")

        else:
            # Compress to one sentence — low confidence
            # Ask LLM to extract the single most useful fact
            compress_prompt = (
                f"Extract the single most relevant fact from this passage "
                f"in one sentence of maximum 20 words:\n\n{chunk['content']}"
            )
            compressed = llm.invoke([
                SystemMessage(content="Extract one key fact. One sentence only."),
                HumanMessage(content=compress_prompt)
            ])
            compressed_text = str(compressed.content).strip()
            section = (
                f"[Passage {i} — relevance {chunk['score']:.2f} — summarised]\n"
                f"{compressed_text}"
            )
            print(f"   Passage {i}: compressed to → {compressed_text[:60]}...")

        context_sections.append(section)

    # ── Job 3: Query-aware system prompt assembly ─────────────
    # Analyse the question type and build matching instructions.
    # A numerical question needs a different answer format
    # than a list question or a risk question.
    #
    # This replaces the generic "you are a document analyst"
    # with instructions that match the specific query intent.

    question = state["question"].lower()

    if any(w in question for w in ["how many", "how much", "what is the", "total", "number"]):
        answer_instruction = (
            "The question asks for a specific number or quantity. "
            "Lead with the number directly. Then explain the context."
        )
    elif any(w in question for w in ["list", "what are", "risks", "factors", "priorities"]):
        answer_instruction = (
            "The question asks for multiple items. "
            "Present as a numbered list. Be concise per item."
        )
    elif any(w in question for w in ["why", "how does", "explain"]):
        answer_instruction = (
            "The question asks for an explanation. "
            "Give a clear, logical explanation in 2-3 sentences."
        )
    else:
        answer_instruction = (
            "Answer clearly and concisely. "
            "Reference specific passages to support your answer."
        )

    # ── Job 4: Token budget tracking ─────────────────────────
    # Rough estimate: 1 token ≈ 4 characters.
    # llama-3.1-8b context window: 8192 tokens.
    # We aim to stay under 6000 tokens for the full prompt
    # (leaving room for the answer itself).

    full_context = "\n\n---\n\n".join(context_sections)
    estimated_tokens = len(full_context) // 4
    TOKEN_WARNING = 5000

    print(f"   Estimated context tokens: ~{estimated_tokens}")

    if estimated_tokens > TOKEN_WARNING:
        print(f"   ⚠️  Context approaching token limit — trimming")
        # Keep only the top 2 chunks if over budget
        full_context = "\n\n---\n\n".join(context_sections[:2])
        estimated_tokens = len(full_context) // 4
        print(f"   Trimmed to ~{estimated_tokens} tokens")

    # ── Assemble the final engineered context ─────────────────
    # This is the complete, optimised package that Answer node
    # will use. Everything above was preparation for this.

    engineered_context = (
        f"ANSWER INSTRUCTION:\n{answer_instruction}\n\n"
        f"DOCUMENT PASSAGES ({len(ranked)} retrieved, "
        f"~{estimated_tokens} tokens):\n\n"
        f"{full_context}"
    )

    print(f"   Context engineered successfully")
    return {"engineered_context": engineered_context}

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
        {state["engineered_context"] if state["engineered_context"] else state["context"]}


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
    graph.add_node("context_engineer",  context_engineer_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("grade",    grade_node)

    # Simple sequential flow —
    # retrieve → answer → grade → done
    graph.add_edge(START,      "retrieve")
    graph.add_edge("retrieve", "context_engineer")
    graph.add_edge("context_engineer", "answer")
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
        engineered_context = "",
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