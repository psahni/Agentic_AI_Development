# Document Intelligence Agent

Ask questions about your own documents. Get answers grounded strictly in what the documents say — no hallucinations, no guessing from LLM memory.

Upload a PDF or text file. Ask a question in plain English. Get a cited, verifiable answer.

---

## How It Works

```
Your document (PDF or TXT)
         ↓
   [Ingestion]  →  Split into chunks  →  Embed  →  Store in ChromaDB
                                                          ↓
Your question   →  [Retriever]  →  Find relevant chunks
                                                          ↓
                   [Agent]      →  LLM answers from those chunks only
                                                          ↓
                   [Grader]     →  Verify answer stays within documents
                                                          ↓
                              Grounded answer with source citations
```

---

## Project Structure

```
document_intelligent_agent/
│
├── core/
│   ├── embeddings.py     converts text to vectors (sentence-transformers)
│   ├── ingestion.py      loads, chunks, and stores documents in ChromaDB
│   ├── retriever.py      finds relevant chunks for a given question
│   └── agent.py          LangGraph agent — retrieve → answer → grade
│
├── docs/                 put your documents here before ingesting
├── chroma_db/            vector database (auto-created on first ingest)
│
├── api.py                FastAPI REST server
├── cli.py                interactive terminal interface
├── start_server.py       starts the API server at port 8000
├── requirements.txt      all Python dependencies
└── LEARNINGS.md          concept-to-code reference guide
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | latest | Agent graph (retrieve → answer → grade nodes) |
| `langchain-groq` | latest | Groq LLM integration |
| `langchain-huggingface` | latest | Local embedding model |
| `langchain-chroma` | latest | ChromaDB vector store integration |
| `langchain-text-splitters` | 1.1.2 | Document chunking |
| `langchain-community` | latest | PDF and text document loaders |
| `sentence-transformers` | 3.3.1 | Local embedding model (`all-MiniLM-L6-v2`) |
| `chromadb` | 0.5.23 | Local vector database |
| `pdfplumber` | 0.11.4 | PDF parsing |
| `groq` | 0.13.1 | Groq API client |
| `fastapi` | 0.115.6 | REST API framework |
| `uvicorn` | 0.32.1 | ASGI server |
| `python-multipart` | 0.0.18 | File upload support |
| `rich` | 13.9.4 | Terminal formatting |
| `typer` | 0.15.1 | CLI framework |
| `python-dotenv` | 1.0.1 | `.env` file loading |
| `pydantic` | 2.10.4 | Data validation |

---

## Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com) — the LLM runs on Groq's servers at no cost

---

## Setup

**1. Clone the repo and enter the project folder**

```bash
git clone https://github.com/psahni/Agentic_AI_Development.git
cd Agentic_AI_Development/document_intelligent_agent
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file in the `document_intelligent_agent/` folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

---

## Quickstart

### Step 1 — Ingest a document

Copy a PDF or `.txt` file into the `docs/` folder, then run:

```bash
python core/ingestion.py
```

This loads the document, splits it into chunks, converts each chunk to a vector, and stores everything in ChromaDB. You only need to do this once per document.

---

### Step 2 — Ask questions

You have two ways to interact with the agent.

#### Option A — Terminal (CLI)

```bash
python cli.py
```

```
==================================================
  DOCUMENT INTELLIGENCE AGENT
  Type your question. Type 'exit' to quit.
==================================================

You: What was the revenue in Q3?

──────────────────────────────────────────────────
Answer:   Revenue in Q3 was $4.2M, a 23% increase
          over Q2. [Source: annual_report.pdf | Chunk 3]
Grounded: True
Sources:  4 chunk(s) used
──────────────────────────────────────────────────
```

#### Option B — REST API

Start the server:

```bash
python start_server.py
```

The server starts at `http://localhost:8000`.

**Check the server is running:**

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "service": "Document Intelligence Agent"}
```

**Ask a question:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue in Q3?"}'
```

```json
{
  "question": "What was the revenue in Q3?",
  "answer": "Revenue in Q3 was $4.2M... [Source: annual_report.pdf | Chunk 3]",
  "grounded": true,
  "sources": 4
}
```

**Interactive API docs** — open in your browser once the server is running:

```
http://localhost:8000/docs
```

---

## What the Response Fields Mean

| Field | Type | Meaning |
|-------|------|---------|
| `question` | string | The question you asked |
| `answer` | string | The agent's answer, with source citations |
| `grounded` | boolean | `true` = answer verified to come from documents only |
| `sources` | integer | Number of document chunks used to form the answer |

If `grounded` is `false`, the answer may contain information from outside your documents. Treat it with caution.

---

## Supported File Types

| Type | Extension | Notes |
|------|-----------|-------|
| PDF | `.pdf` | Text-based PDFs only. Scanned image PDFs are not supported. |
| Plain text | `.txt` | UTF-8 encoded |

---

## Troubleshooting

**`GROQ_API_KEY is not set`**
Create a `.env` file in the project folder with `GROQ_API_KEY=your_key_here`.

**`No relevant chunks found above threshold`**
The question may not match any content in your documents. Try rephrasing, or check that ingestion completed successfully.

**`chroma_db/` folder is missing**
Run ingestion first (`python core/ingestion.py`). ChromaDB creates this folder automatically on first run.

**Pylance shows import errors for langchain packages**
Make sure your VS Code Python interpreter points to the virtual environment where you ran `pip install`. Use `Ctrl+Shift+P` → **Python: Select Interpreter**.
