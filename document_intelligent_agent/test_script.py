# test_rag.py
#
# End-to-end RAG test script.
# Runs the complete pipeline in one go:
# Ingest → Embed → Store → Retrieve → Answer

import shutil
import os

print("=" * 55)
print("  RAG PIPELINE — END TO END TEST")
print("=" * 55)

# ── Step 1: Clean slate ───────────────────────────────────
# Delete existing ChromaDB so we start fresh every test run
print("\n[Step 1] Clearing old vector database...")
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("   Old ChromaDB cleared")
else:
    print("   No existing database found — starting fresh")

# ── Step 2: Ingest document ───────────────────────────────
# Load the document, split into chunks, embed, store
print("\n[Step 2] Ingesting document...")
from core.ingestion import ingest_document
ingest_document("docs/sample.txt")

# ── Step 3: Verify what was stored ───────────────────────
# Connect to ChromaDB and count stored chunks
print("\n[Step 3] Verifying vector database...")
import chromadb
client     = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("documents")
count      = collection.count()
print(f"   Chunks stored in ChromaDB: {count}")

# ── Step 4: Ask questions ─────────────────────────────────
print("\n[Step 4] Running test questions...")
from core.agent import ask

questions = [
    "What is the total revenue of ACME Corporation in 2024?",
    "What are the three risk factors for 2025?",
    "What is the stock price of ACME Corporation?"  # not in document
]

print("\n" + "=" * 55)

for i, question in enumerate(questions, 1):
    print(f"\nQuestion {i}: {question}")
    print("-" * 55)

    result = ask(question)

    print(f"Answer:   {result['answer']}")
    print(f"Grounded: {result['grounded']}")
    print(f"Sources:  {result['sources']} chunk(s) used")
    print("=" * 55)

print("\nAll tests complete.")