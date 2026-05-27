# main.py
#
# Entry point for the Document Intelligence Agent.
# Handles two things:
# 1. Ingestion — indexes documents into ChromaDB
# 2. Interface — starts CLI or REST API
#
# Usage:
#   python main.py              → starts CLI
#   python main.py --api        → starts REST API server
#   python main.py --ingest     → re-ingests all docs in docs/
#   python main.py --ingest --api → ingest then start API

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_PATH    = PROJECT_ROOT / "docs"
CHROMA_PATH  = PROJECT_ROOT / "chroma_db"


def run_ingestion():
    from core.ingestion import ingest_document

    docs = list(DOCS_PATH.glob("*.txt")) + list(DOCS_PATH.glob("*.pdf"))

    if not docs:
        print(f"No documents found in {DOCS_PATH}/")
        print("Add .txt or .pdf files to the docs/ folder and try again.")
        return False

    print(f"\nFound {len(docs)} document(s) to ingest:")
    for doc in docs:
        print(f"  - {doc.name}")

    print()
    for doc in docs:
        ingest_document(str(doc))

    print("\nAll documents ingested successfully.")
    return True


def check_chroma_exists():
    # If ChromaDB folder exists and has content,
    # documents have already been ingested.
    # User does not need to re-ingest on every run.
    return CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir())


def run_cli():
    from cli import run_cli as start_cli
    start_cli()


def run_api():
    import uvicorn
    print("\nStarting REST API server...")
    print("Docs available at: http://localhost:8000/docs")
    print("Health check:      http://localhost:8000/health")
    print("Ask endpoint:      POST http://localhost:8000/ask")
    print("\nPress Ctrl+C to stop.\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


def main():
    parser = argparse.ArgumentParser(
        description="Document Intelligence Agent"
    )
    parser.add_argument(
        "--ingest",
        action  = "store_true",
        help    = "Ingest documents from docs/ folder into ChromaDB"
    )
    parser.add_argument(
        "--api",
        action  = "store_true",
        help    = "Start REST API server instead of CLI"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  DOCUMENT INTELLIGENCE AGENT")
    print("=" * 55)

    # ── Ingestion ─────────────────────────────────────────────
    if args.ingest:
        print("\nIngestion mode selected.")
        success = run_ingestion()
        if not success:
            return

    elif not check_chroma_exists():
        # First run — no ChromaDB yet — auto-ingest
        print("\nNo vector database found. Running ingestion first...")
        success = run_ingestion()
        if not success:
            return

    else:
        print("\nVector database found. Skipping ingestion.")
        print("Use --ingest flag to re-index documents.")

    # ── Interface ─────────────────────────────────────────────
    print()
    if args.api:
        run_api()
    else:
        run_cli()

def warmup():
    print("Warming up embedding model...")
    from core.embeddings import get_embedding_model
    model = get_embedding_model()
    # Run one dummy embedding to fully initialise
    model.embed_query("warmup")
    print("Model warm. Ready for traffic.")

if __name__ == "__main__":
    warmup() # Needed for production to tackle cold start. Other sol - Combine eager loading + keepalive
    main()   # eager loading means loading the modal at the time of import module - so server start will take some more time