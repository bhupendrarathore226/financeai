"""
Vector store abstraction for FinanceAI.

Responsibility
--------------
This module owns the lifecycle of two heavyweight external resources:

  1. **ChromaDB client** — the embedded vector database that persists
     transaction embeddings to disk (at `config.CHROMA_PATH`).
  2. **SentenceTransformer model** — the local ML model that converts raw
     text strings into numeric vectors (embeddings).

Both resources are expensive to initialise (disk I/O and model loading),
so this module implements the **lazy singleton** pattern:
  - The object is created only the first time it is requested.
  - Every subsequent call returns the already-created instance immediately.
  - Nothing is imported or initialised at module load time, which keeps
    startup fast and makes unit testing easier (tests can patch before first use).

Why lazy singletons matter for testing
---------------------------------------
If ChromaDB and SentenceTransformer were imported and initialised at the
top of the file, every test that imports ANY module in the services package
would trigger real database and ML-model setup, making tests slow and
fragile.  By deferring initialisation to the first function call, tests can
set `sys.modules["services.store"]` to a lightweight fake *before* the
real module is imported (see tests/test_api.py and tests/test_ingest.py).
"""

from typing import Any  # Used for return-type hints when the concrete type is internal

import core.config as config  # Centralised settings (paths, model names, etc.)


# ---------------------------------------------------------------------------
# Module-level singleton cache variables
# ---------------------------------------------------------------------------
# These variables hold the single shared instance of each resource.
# They start as None and are populated on first use inside the factory functions.
# Using module-level variables (rather than class attributes) keeps the code
# simple while still achieving the singleton behaviour.

_embedding_model = None  # Will hold a SentenceTransformer instance once initialised
_client = None           # Will hold a chromadb.PersistentClient instance once initialised


# ---------------------------------------------------------------------------
# Private helper: ChromaDB client factory
# ---------------------------------------------------------------------------

def _get_client() -> Any:
    """
    Return the shared ChromaDB persistent client, creating it on first call.

    ChromaDB is an embedded vector database.  "Persistent" means it saves
    its data to a directory on disk (config.CHROMA_PATH) so the data survives
    server restarts.  If the directory does not exist ChromaDB creates it.

    This function is private (prefixed with _) because external code should
    always go through `get_collection()` rather than holding a raw client.

    Returns
    -------
    chromadb.PersistentClient
        The shared database client instance.

    Raises
    ------
    RuntimeError
        If `chromadb` is not installed in the current Python environment.
    """
    global _client  # Tell Python we want to modify the module-level variable

    if _client is None:
        # Only import chromadb here (not at the top of the file) so that the
        # module can be imported in environments where chromadb is absent
        # (e.g. lightweight CI runners or test environments using stubs).
        try:
            import chromadb
        except Exception as exc:
            raise RuntimeError("ChromaDB is not available in this environment") from exc

        # PersistentClient stores embeddings on disk at the given path.
        # Subsequent calls will load the existing database from that path.
        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    return _client


# ---------------------------------------------------------------------------
# Public factory: embedding model
# ---------------------------------------------------------------------------

def get_embedding_model() -> Any:
    """
    Return the shared SentenceTransformer embedding model, loading it on first call.

    SentenceTransformer converts a list of text strings into a 2-D NumPy array
    of floating-point vectors.  Each vector captures the semantic meaning of
    the text, so similar sentences produce vectors that are close together in
    high-dimensional space.

    The model specified by config.EMBED_MODEL ("all-MiniLM-L6-v2") is a small,
    fast model that works well on CPU.  The first call downloads the model
    weights from Hugging Face (~90 MB); subsequent calls load them from the
    local cache instantly.

    Returns
    -------
    sentence_transformers.SentenceTransformer
        The shared model instance, ready to call `.encode(list_of_strings)`.

    Raises
    ------
    RuntimeError
        If `sentence_transformers` is not installed.
    """
    global _embedding_model  # Modify the module-level singleton variable

    if _embedding_model is None:
        # Deferred import for the same reason as chromadb above.
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("SentenceTransformer is not available in this environment") from exc

        # Loading the model reads the weights from disk (or downloads them).
        # After this line, _embedding_model.encode(texts) can be called.
        _embedding_model = SentenceTransformer(config.EMBED_MODEL)

    return _embedding_model


# ---------------------------------------------------------------------------
# Public factory: ChromaDB collection
# ---------------------------------------------------------------------------

def get_collection():
    """
    Return (or create) the ChromaDB collection used to store transaction embeddings.

    A ChromaDB "collection" is analogous to a table in a relational database.
    It stores:
      - `documents`  : The raw text strings (one per transaction row).
      - `embeddings` : The numeric vector representation of each document.
      - `ids`        : A unique string identifier for each record.
      - `metadatas`  : A dict of arbitrary key-value metadata per record
                       (used here to store source filename and chunk index).

    `get_or_create_collection` is idempotent — it returns the existing
    collection if it already exists, or creates a new empty one if it does not.
    This means the server can restart safely without losing data.

    Returns
    -------
    chromadb.Collection
        The collection instance, ready for `.add()`, `.query()`, and `.get()`.
    """
    # Obtain (or lazily initialise) the client, then get/create the collection.
    client = _get_client()
    return _client.get_or_create_collection(name=config.COLLECTION_NAME)
