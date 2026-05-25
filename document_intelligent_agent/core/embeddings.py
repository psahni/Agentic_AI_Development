# core/embeddings.py
#
# PURPOSE:
# This file sets up the embedding model — the engine
# that converts text into numbers (vectors).
#
# It is isolated here so every other file imports from
# one place. Swap the model here and the entire system
# uses the new model automatically.
#
# WHY sentence-transformers/all-MiniLM-L6-v2?
# - Runs completely locally — no API calls, no cost
# - Small (80MB) and fast
# - Produces 384-dimensional vectors
# - Strong performance for semantic similarity tasks
# - Industry standard for local RAG development
#
# In production you might use:
# - OpenAI text-embedding-3-small (better quality, costs money)
# - Cohere embed-v3 (multilingual, costs money)
# But for learning, local is better — no surprises, no bills

from langchain_huggingface import HuggingFaceEmbeddings

# Module-level singleton — the model loads once when
# this module is first imported and stays in memory.
# Loading a model takes 2-3 seconds. We do not want
# to reload it on every function call.
_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    # WHY a singleton pattern here?
    # The embedding model is heavy — it loads a neural
    # network into memory. Creating it once and reusing
    # it is a standard performance pattern.
    global _embedding_model

    if _embedding_model is None:
        print("   Loading embedding model (first time only)...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name      = "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs    = {"device": "cpu"},
            encode_kwargs   = {"normalize_embeddings": True}
        )
        print("   Embedding model ready")

    return _embedding_model


def embed_text(text: str) -> list[float]:
    # Convert a single piece of text into a vector.
    # Returns a list of 384 floats.
    # Used when embedding a search query at retrieval time.
    model = get_embedding_model()
    return model.embed_query(text)



# python -c "
# from core.embeddings import embed_text
# vector = embed_text('The company revenue grew by 23 percent')
# print(f'Vector dimensions: {len(vector)}')
# print(f'First 5 numbers: {vector[:5]}')
# print('Embeddings working correctly')
# "