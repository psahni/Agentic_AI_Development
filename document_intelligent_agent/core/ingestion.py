# core/ingestion.py
#
# PURPOSE:
# Loads a document, splits it into chunks, embeds each
# chunk, and stores everything in ChromaDB.
#
# Run this once per document. After it runs, your
# document is fully indexed and searchable.
#
# THE THREE STAGES:
# 1. Load   → read file from disk into memory
# 2. Chunk  → split into overlapping pieces
# 3. Store  → embed + save to ChromaDB

from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from core.embeddings import get_embedding_model

CHROMA_PATH = "chroma_db"


def load_document(filepath: str):
    # WHY two different loaders?
    # PDFs are binary files — you cannot read them with
    # open(). PyPDFLoader handles the binary parsing.
    # Text files are plain strings — TextLoader is enough.
    path = Path(filepath)
    print(f"\n[Ingestion] Loading: {path.name}")

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {filepath}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    print(f"   Loaded {len(docs)} page(s)")
    return docs


def chunk_document(documents):
    # WHY RecursiveCharacterTextSplitter?
    # It tries to split on natural boundaries in this order:
    # paragraphs first → then sentences → then words → then characters
    # This keeps related sentences together in the same chunk.
    #
    # chunk_size = 500 characters per chunk
    # A chunk of 500 chars is roughly 80-100 words —
    # enough context for meaningful retrieval without noise.
    #
    # chunk_overlap = 100 characters
    # WHY overlap? Imagine a key insight spans the boundary
    # between two chunks. Without overlap, it gets split and
    # neither chunk contains the full idea. Overlap ensures
    # boundary-spanning content appears in at least one chunk.

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 100,
        separators    = ["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"   Split into {len(chunks)} chunks")

    # Show the first chunk so you can see what chunking produces
    if chunks:
        print(f"\n   First chunk preview:")
        print(f"   '{chunks[0].page_content[:200]}...'")

    return chunks


def store_in_chroma(chunks, collection_name: str = "documents"):
    import shutil
    # WHY collection_name?
    # ChromaDB organises vectors into collections —
    # like tables in a database. Each document collection
    # stays separate. You could have "financial_reports"
    # and "contracts" as separate collections.
    #
    # Chroma.from_documents() does three things at once:
    # 1. Takes each chunk's text
    # 2. Calls our embedding model to convert it to a vector
    # 3. Stores both the vector AND the original text in ChromaDB
    #
    # This is important — ChromaDB stores BOTH the numbers
    # (for searching) AND the original text (for returning
    # to the LLM). You get back human-readable text,
    # not just numbers.
    # WHY delete before re-creating?
    # ChromaDB appends to existing collections.
    # If we do not clear first, every re-ingest
    # duplicates all chunks already in the database.
    # Deleting and recreating guarantees a clean state.
    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)
        print("   Cleared existing ChromaDB")


    print(f"\n   Embedding and storing {len(chunks)} chunks...")

    vectorstore = Chroma.from_documents(
        documents       = chunks,
        embedding       = get_embedding_model(),
        persist_directory = CHROMA_PATH,
        collection_name = collection_name
    )

    print(f"   Stored in ChromaDB at: {CHROMA_PATH}/")
    return vectorstore


def ingest_document(filepath: str, collection_name: str = "documents"):
    # Main entry point — runs all three stages in sequence.
    # Call this once for each document you want to index.
    print("=" * 50)
    print("  DOCUMENT INGESTION")
    print("=" * 50)

    documents = load_document(filepath)
    chunks    = chunk_document(documents)
    store     = store_in_chroma(chunks, collection_name)

    print(f"\n   ✅ Ingestion complete")
    print(f"   Document is now searchable")
    print("=" * 50)

    return store


if __name__ == "__main__":
    # Quick test — run this file directly to ingest
    # the sample document we created earlier
    ingest_document("docs/sample.txt")

    