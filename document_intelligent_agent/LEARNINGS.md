# Document Intelligence Agent — What You Built and What It Taught You

Each section below ties a concept directly to the code that implements it.
Read the concept, then look at the file — you will see exactly where the idea lives.

---

## 1. RAG Pipeline — How Documents Become Answers

**The idea.**
RAG stands for Retrieval-Augmented Generation. Instead of asking an LLM to answer from memory, you first retrieve relevant text from your own documents, then hand that text to the LLM as context. The LLM answers only from what you gave it.

**The three-stage flow you built.**

```
Document on disk
      ↓
  [ingestion.py]  →  Load → Chunk → Embed → Store in ChromaDB
                                                    ↓
User question  →  [retriever.py]  →  Find closest chunks
                                                    ↓
                  [agent.py]      →  LLM answers from those chunks only
```

**Where it lives in the code.**

| Stage | File | Function |
|-------|------|----------|
| Load | `core/ingestion.py` | `load_document()` |
| Chunk | `core/ingestion.py` | `chunk_document()` |
| Store | `core/ingestion.py` | `store_in_chroma()` |
| Retrieve | `core/retriever.py` | `retrieve()` |
| Answer | `core/agent.py` | `answer_node()` |

**Why it matters.**
Without retrieval, the LLM answers from training data — which may be outdated, wrong, or simply not your data. RAG makes the LLM answer from *your* documents, making every answer verifiable.

---

## 2. Embeddings — Text Converted to Numbers That Capture Meaning

**The idea.**
An embedding model reads a sentence and outputs a list of numbers — a vector. The key insight is that sentences with similar *meanings* produce vectors that are mathematically close to each other, even if the words are completely different.

> "Revenue increased by 23%" and "Sales grew significantly" will have similar vectors, even though they share no words.

**Where it lives in the code.**

**File:** `core/embeddings.py` — `get_embedding_model()` and `embed_text()`

```python
# Loads once, stays in memory — the singleton pattern
_embedding_model = HuggingFaceEmbeddings(
    model_name    = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True}
)
```

The model used (`all-MiniLM-L6-v2`) converts any text into **384 numbers**. Every chunk stored in ChromaDB has 384 numbers beside it. Every question you ask also becomes 384 numbers. Retrieval is just finding which stored numbers are closest to the question numbers.

**The singleton pattern.**
The model takes 2–3 seconds to load. The `_embedding_model = None` guard at the top of `embeddings.py` ensures it loads exactly once and is reused for every subsequent call — both during ingestion and retrieval.

---

## 3. Chunking — Why Splitting Matters More Than You Think

**The idea.**
A full document can be thousands of words. An LLM context window is limited. More importantly, if you send the entire document as context, the LLM cannot distinguish what is relevant from what is noise. Chunking solves this by splitting the document into small, focused pieces so retrieval can return only the most relevant ones.

**Where it lives in the code.**

**File:** `core/ingestion.py` — `chunk_document()`

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 500,   # ~80-100 words per chunk
    chunk_overlap = 100,   # 100 characters shared with the next chunk
    separators    = ["\n\n", "\n", ". ", " ", ""]
)
```

**Three decisions made here and why.**

| Decision | Value | Reason |
|----------|-------|--------|
| `chunk_size` | 500 characters | Enough context to hold one complete idea without noise |
| `chunk_overlap` | 100 characters | Key insights that span a boundary appear in at least one chunk |
| `separators` | Paragraph → sentence → word → character | Splits on natural language boundaries, keeping related sentences together |

**Why overlap exists.**
Imagine the most important sentence in a document falls exactly at the boundary between chunk 4 and chunk 5. Without overlap, it is split in half — neither chunk contains the full idea and retrieval misses it. Overlap ensures boundary-spanning content is always captured whole.

---

## 4. ChromaDB — The Vector Database

**The idea.**
ChromaDB is a database designed specifically for vectors. Unlike a regular database that stores rows and columns, ChromaDB stores each chunk as a pair: the original text *and* its vector. When you search, it compares your question's vector against all stored vectors and returns the closest matches.

**Where it lives in the code.**

**Storing** — `core/ingestion.py` — `store_in_chroma()`
```python
vectorstore = Chroma.from_documents(
    documents         = chunks,
    embedding         = get_embedding_model(),
    persist_directory = CHROMA_PATH,       # saves to chroma_db/ folder on disk
    collection_name   = collection_name
)
```

**Reading** — `core/retriever.py` — `get_vectorstore()`
```python
return Chroma(
    persist_directory  = CHROMA_PATH,
    embedding_function = get_embedding_model(),
    collection_name    = collection_name
)
```

**What ChromaDB holds for each chunk.**

```
┌─────────────────────────────────────────────────────┐
│  Original text  │  Vector (384 numbers)  │  Metadata │
│  "Revenue grew  │  [0.23, -0.11, 0.87,  │  source,  │
│   by 23%..."    │   0.04, ...]           │  page     │
└─────────────────────────────────────────────────────┘
```

You get back human-readable text from a mathematical search — that is the core value of a vector database.

---

## 5. Grounding and Citation — Forcing Answers to Stay Inside the Documents

**The idea.**
Left to itself, an LLM will blend retrieved content with its own training memory. The answer might be half-correct — some from your document, some invented. Grounding means instructing the LLM to use *only* what it was given, and citation means labeling exactly which chunk each claim came from.

**Where it lives in the code.**

**Grounding instruction** — `core/agent.py` — `answer_node()` system prompt:
```
STRICT RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say exactly:
   "This information is not available in the provided documents."
3. Never use your own training knowledge
```

**Citation labels** — `core/retriever.py` — `format_context()`
```python
section = (
    f"[Source: {source} | Chunk {i} | Relevance: {chunk['score']}]\n"
    f"{chunk['content']}"
)
```

Every chunk handed to the LLM carries a label. The LLM sees `[Source: annual_report.pdf | Chunk 2]` before the text. When it writes its answer, it knows — and is instructed — to include this label so every claim can be traced back to a specific chunk in a specific document.

---

## 6. Grade Node — The Anti-Hallucination Guard

**The idea.**
Even strict instructions are not enough. LLMs occasionally blend retrieved content with training memory. The grade node is a second, independent LLM call whose only job is to verify that the first answer stayed within the bounds of the retrieved chunks.

**Where it lives in the code.**

**File:** `core/agent.py` — `grade_node()`

```python
grade_prompt = f"""
DOCUMENT EXCERPTS:
{plain_chunks}

ANSWER TO CHECK:
{state["answer"]}

Does the answer contain ONLY facts that appear
in the document excerpts above?
Reply with exactly one word: YES or NO
"""
```

**The flow.**

```
answer_node generates answer
        ↓
grade_node reads the same chunks + the answer
        ↓
   YES → answer is grounded → return to user
    NO → answer flagged     → warn user
```

**The edge case it handles.**
If the answer says *"This information is not available in the provided documents"*, that is a perfectly grounded response — the agent correctly stayed within bounds by admitting it does not know. The grader recognises this and marks it `grounded: True`.

---

## 7. Decoupling — One Core, Two Entry Points

**The idea.**
Business logic and interface logic should never know about each other. The RAG pipeline does not care whether the question arrived from a terminal command or an HTTP request. `cli.py` and `api.py` do not care how retrieval or grounding works. They all meet at one clean function.

**Where it lives in the code.**

**The single meeting point** — `core/agent.py` — `ask()`
```python
def ask(question: str, collection_name: str = "documents") -> dict:
    # cli.py calls this. api.py calls this.
    # Neither knows anything about LangGraph, ChromaDB, or Groq.
    result = _graph.invoke(initial_state)
    return {
        "question": result["question"],
        "answer":   result["answer"],
        "grounded": result["grounded"],
        "sources":  len(result["chunks"])
    }
```

**How the two doors use it.**

```
Terminal user
     ↓
  cli.py  ──────┐
                ├──→  core/agent.ask()  →  LangGraph  →  ChromaDB + Groq
  api.py  ──────┘
     ↑
HTTP request
```

Changing the LLM, the vector database, or the graph structure requires touching only `core/`. The CLI and API need zero changes. This is the architectural benefit of decoupling — interfaces are stable, internals can evolve freely.

---

## 8. FastAPI — Exposing the Agent as a REST Service

**The idea.**
A Python function running on your laptop is useful only to you. Wrapping it in a FastAPI server makes it accessible to any application, any language, any team — over standard HTTP. FastAPI adds input validation, error handling, and auto-generated documentation for free.

**Where it lives in the code.**

**File:** `api.py`

```python
app = FastAPI(
    title       = "Document Intelligence Agent",
    description = "Answers grounded in retrieved content only.",
    version     = "1.0.0"
)
```

**The two endpoints.**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Monitoring systems ping this to confirm the server is alive |
| `/ask` | POST | Accepts a question, returns a grounded answer |

**Input validation — Pydantic models.**
```python
class QuestionRequest(BaseModel):
    question:        str
    collection_name: str = "documents"
```
If a request arrives without a `question` field, FastAPI rejects it automatically before your code runs. No manual validation needed.

**Error isolation — the try/except in `ask_question()`.**
If the agent crashes on one request — retrieval fails, LLM times out — the `except` block returns an HTTP 500 response. The server keeps running and handles the next request. Without this, one bad request would crash the entire process.

**Free documentation.**
Run the server and open `http://localhost:8000/docs` — FastAPI generates an interactive API explorer from your code with no extra work.

---

## Summary Map

| Concept | File | Key Function / Class |
|---------|------|----------------------|
| RAG pipeline (full flow) | `core/ingestion.py` → `core/retriever.py` → `core/agent.py` | `ingest_document()` → `retrieve()` → `answer_node()` |
| Embeddings | `core/embeddings.py` | `get_embedding_model()`, `embed_text()` |
| Chunking | `core/ingestion.py` | `chunk_document()` |
| Vector storage and search | `core/ingestion.py`, `core/retriever.py` | `store_in_chroma()`, `get_vectorstore()` |
| Grounding and citation | `core/agent.py`, `core/retriever.py` | `answer_node()` system prompt, `format_context()` |
| Anti-hallucination guard | `core/agent.py` | `grade_node()` |
| Decoupling (one core, two doors) | `core/agent.py` | `ask()` |
| REST API | `api.py` | `QuestionRequest`, `ask_question()` |
