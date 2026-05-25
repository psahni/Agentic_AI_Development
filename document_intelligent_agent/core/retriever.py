# core/retriever.py
#
# PURPOSE:
# Searches ChromaDB for chunks relevant to a question.
# This is the search engine of the RAG system.
#
# HOW SEMANTIC SEARCH WORKS HERE:
# 1. Your question arrives as plain text
# 2. Embedding model converts it to a 384-number vector
# 3. ChromaDB compares that vector to every stored chunk vector
# 4. Returns the chunks with the smallest mathematical distance
#    — these are the most semantically similar chunks
#
# WHY IS THIS BETTER THAN KEYWORD SEARCH?
# Keyword: "Singapore revenue" finds only chunks
#          containing those exact words
# Semantic: "Singapore revenue" also finds chunks about
#           "Asia Pacific growth" and "new market expansion"
#           because they mean similar things mathematically

from langchain_chroma import Chroma
from core.embeddings import get_embedding_model

CHROMA_PATH = "chroma_db"


def get_vectorstore(collection_name: str = "documents") -> Chroma:
    # Connect to existing ChromaDB collection.
    # WHY not create a new one here?
    # Ingestion already created and populated the collection.
    # Here we just open a connection to read from it.
    # Think of it like opening a database connection
    # vs creating the database — two different operations.
    return Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = get_embedding_model(),
        collection_name    = collection_name
    )


def retrieve(question: str, k: int = 4,
             collection_name: str = "documents") -> list[dict]:
    # Main retrieval function.
    # Takes a question, returns a list of relevant chunks
    # with their content and source metadata.
    #
    # similarity_search_with_relevance_scores returns
    # each chunk paired with a score between 0 and 1.
    # Score of 1.0 = perfect match
    # Score below 0.5 = probably not relevant
    # We filter out anything below 0.3 to avoid
    # returning completely irrelevant chunks.

    print(f"\n[Retriever] Searching for: '{question}'")

    vectorstore = get_vectorstore(collection_name)

    results = vectorstore.similarity_search_with_relevance_scores(
        question, k=k
    )

    # Filter low-relevance results
    filtered = [
        (doc, score)
        for doc, score in results
        if score >= 0.3
    ]

    if not filtered:
        print("   No relevant chunks found above threshold")
        return []

    print(f"   Found {len(filtered)} relevant chunks:")
    for i, (doc, score) in enumerate(filtered, 1):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"   {i}. Score {score:.2f} — '{preview}...'")

    # Return clean list of dicts — easy to work with
    # in the agent and easy to serialize for the API
    return [
        {
            "content":  doc.page_content,
            "score":    round(score, 3),
            "source":   doc.metadata.get("source", "unknown"),
            "page":     doc.metadata.get("page", 0)
        }
        for doc, score in filtered
    ]


def format_context(chunks: list[dict]) -> str:
    # Formats retrieved chunks into a clean string
    # that gets injected into the LLM prompt.
    #
    # WHY format with SOURCE labels?
    # This is citation enforcement in action.
    # When the LLM sees [Source: sample.txt | Chunk 1],
    # it knows where each piece of information came from.
    # We instruct the LLM to include these labels in its
    # answer — making every claim traceable.
    if not chunks:
        return "No relevant information found in the documents."

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        source  = Path(chunk["source"]).name if chunk["source"] != "unknown" else "document"
        section = (
            f"[Source: {source} | Chunk {i} | "
            f"Relevance: {chunk['score']}]\n"
            f"{chunk['content']}"
        )
        formatted.append(section)

    return "\n\n---\n\n".join(formatted)


# Need Path for format_context
from pathlib import Path

