"""
Query service for FinanceAI  —  semantic search + LLM answer generation.

Responsibility
--------------
This module implements the **RAG (Retrieval-Augmented Generation)** pipeline,
which is the core intelligence of the FinanceAI chatbot.  Given a user's
natural-language question, it:

  1. **Retrieves** the most relevant transaction records from ChromaDB using
     vector similarity search (semantic_search).
  2. **Augments** a prompt with those retrieved transactions as context.
  3. **Generates** a grounded, factual answer by calling the OpenAI chat API.

Why RAG instead of just asking the LLM directly?
-------------------------------------------------
LLMs like GPT-4o have a fixed context window and no access to your personal
bank data.  RAG solves both problems: we store your transactions as vectors
in ChromaDB, search for the relevant ones at query time, and inject them into
the LLM's prompt.  The LLM then reasons over real personal data instead of
making things up (hallucinating).

Flow diagram
------------
  User question (text)
      │
      ├─ get_embedding_model().encode(question)  →  question vector
      │
      ├─ ChromaDB.query(question_vector)         →  top-K transaction strings
      │
      ├─ _build_prompt(question, transactions)   →  full prompt string
      │
      └─ OpenAI.chat.completions.create(prompt)  →  natural-language answer
"""

import logging  # Standard library: structured log messages

# OpenAI Python SDK — used to call the OpenAI chat completions API.
# The SDK handles authentication, HTTP retries, and response parsing.
from openai import OpenAI

import core.config as config  # Centralised configuration values

# Lazy singleton factories for ChromaDB and the embedding model.
# Using factories ensures the heavy objects are only initialised once.
from services.store import get_collection, get_embedding_model


# Module-level logger.  Log records appear as "services.query" in output.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception type
# ---------------------------------------------------------------------------

class QueryError(Exception):
    """
    Raised when a known, recoverable error prevents a question from being answered.

    Examples include:
      - An empty or whitespace-only question was submitted.
      - The embedding model failed to process the question.
      - ChromaDB could not be queried.
      - The OpenAI API returned an error.

    The API layer (api/main.py) catches this exception and maps it to an
    HTTP 400 response, distinguishing it from unexpected server errors (500).
    """
    pass  # No extra attributes needed; the message string carries the details.


# ---------------------------------------------------------------------------
# Stage 1: Semantic search
# ---------------------------------------------------------------------------

def semantic_search(question: str, top_k: int = config.DEFAULT_TOP_K) -> list[str]:
    """
    Find the `top_k` transaction strings most semantically similar to `question`.

    How vector similarity search works
    ------------------------------------
    1. The question is converted into an embedding vector using the same
       SentenceTransformer model that was used when ingesting transactions.
    2. ChromaDB computes the cosine distance between the question vector and
       every stored transaction vector.
    3. The `top_k` transactions with the smallest distance (most similar
       meaning) are returned as plain strings.

    This approach finds relevant transactions even when the user's wording
    does not exactly match the text in the PDF.  For example, "dining out"
    will match "Restaurant — Olive Garden" because they are semantically close.

    Parameters
    ----------
    question : str
        The natural-language question to search for.  Must be non-empty.
    top_k : int, optional
        Number of most-relevant transactions to return.  Defaults to
        config.DEFAULT_TOP_K (5).  Higher values give the LLM more context
        but also increase prompt size and therefore cost.

    Returns
    -------
    list[str]
        A list of up to `top_k` transaction strings from ChromaDB, ordered
        from most to least similar to the question.  Returns an empty list
        if ChromaDB contains no documents yet.

    Raises
    ------
    QueryError
        If the question is empty, embedding fails, or ChromaDB cannot be queried.
    """

    # Reject empty questions early to avoid wasting model inference and tokens.
    if not question or not question.strip():
        raise QueryError("Question cannot be empty")

    # --- Embed the question ---
    # The embedding must use the SAME model as was used during ingestion;
    # otherwise the question vector and transaction vectors live in different
    # spaces and similarity scores are meaningless.
    try:
        embedding_model = get_embedding_model()
        # encode() expects a list of strings; we wrap the single question in [].
        # .tolist() converts the NumPy array to a Python list for ChromaDB.
        q_embed = embedding_model.encode([question]).tolist()
    except Exception as exc:
        logger.exception("Failed to embed question")
        raise QueryError("Could not prepare search query") from exc

    # --- Query ChromaDB for similar vectors ---
    try:
        collection = get_collection()
        # query_embeddings: the vector(s) to search for (one per query).
        # n_results: how many nearest neighbours to return.
        results = collection.query(query_embeddings=q_embed, n_results=top_k)
    except Exception as exc:
        logger.exception("Semantic search failed")
        raise QueryError("Could not query vector database") from exc

    # ChromaDB returns results as a nested list: results["documents"][query_index].
    # We sent one query, so we always want index [0].
    documents = results.get("documents", [])
    if not documents or not documents[0]:
        return []  # No transactions stored yet, or none matched the query

    return documents[0]  # List of matched transaction strings for query 0


# ---------------------------------------------------------------------------
# OpenAI client singleton
# ---------------------------------------------------------------------------

# Module-level cache for the OpenAI client.
# Initialised lazily on first call to get_openai_client() to avoid failing
# at import time if OPENAI_API_KEY is not set.
_openai_client = None


def get_openai_client() -> OpenAI:
    """
    Return the shared OpenAI client, creating it on first call.

    The OpenAI client object handles authentication (via the API key),
    HTTP connection pooling, and automatic retries.  Creating it once and
    reusing it is more efficient than instantiating it on every request.

    Returns
    -------
    openai.OpenAI
        The shared client instance, ready to make API calls.
    """
    global _openai_client  # Modify the module-level variable

    if _openai_client is None:
        # config.OPENAI_API_KEY is read from the OPENAI_API_KEY environment variable.
        # If not set, OpenAI will raise an AuthenticationError on the first API call.
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

    return _openai_client


# ---------------------------------------------------------------------------
# Stage 2: Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Assemble the final prompt string that will be sent to the OpenAI chat API.

    Prompt engineering decisions
    -----------------------------
    - Transactions are presented as a bullet list (one per line) so the LLM
      can clearly distinguish individual records.
    - The instruction asks the LLM to admit uncertainty ("say you are not sure")
      rather than hallucinate an answer when the context is insufficient.
    - The question appears at the end so the LLM reads the context first,
      which tends to produce more grounded answers.

    Parameters
    ----------
    question : str
        The user's original question.
    context_chunks : list[str]
        Transaction strings retrieved from semantic search.  May be empty
        if no transactions are stored yet.

    Returns
    -------
    str
        The complete prompt string ready to pass to the OpenAI chat completions
        endpoint as the `content` of a "user" message.
    """

    if context_chunks:
        # Format each transaction as a bullet point for readability.
        # Keep context compact and deterministic for stable, reproducible prompting.
        context_text = "\n".join(f"- {chunk}" for chunk in context_chunks)
    else:
        # Explicitly tell the LLM there is no data so it doesn't make things up.
        context_text = "No transaction context found."

    return (
        "Answer the question using these transactions. "
        "If the answer cannot be inferred, say you are not sure.\n\n"
        f"Transactions:\n{context_text}\n\n"
        f"Question: {question}"
    )


# ---------------------------------------------------------------------------
# Stage 3: LLM answer generation (public entry point for /chat)
# ---------------------------------------------------------------------------

def ask_llm(question: str) -> str:
    """
    Run the full RAG pipeline and return a natural-language answer.

    This is the main function called by the /chat API endpoint.  It
    combines semantic_search and _build_prompt, then sends the assembled
    prompt to OpenAI and returns the generated text.

    Parameters
    ----------
    question : str
        The user's natural-language question (already stripped of whitespace
        by the caller in api/main.py).

    Returns
    -------
    str
        The LLM-generated answer as a plain string.  Never None (falls back
        to empty string if the model returns no content).

    Raises
    ------
    QueryError
        Propagated from semantic_search if embedding or ChromaDB fails.
        Also raised directly if the OpenAI API call fails.
    """

    # Retrieve the most relevant transactions from ChromaDB.
    context_chunks = semantic_search(question)

    # Build the prompt that will be sent to OpenAI.
    prompt = _build_prompt(question, context_chunks)

    # --- Call the OpenAI chat completions API ---
    try:
        response = get_openai_client().chat.completions.create(
            model=config.OPENAI_MODEL,
            # "messages" is a list of turn objects.  Using a single "user" message
            # is the simplest format; more complex conversations would alternate
            # "user" and "assistant" turns.
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.exception("OpenAI completion failed")
        raise QueryError("Could not generate answer from language model") from exc

    # response.choices[0].message.content holds the generated text.
    # The `or ""` guard handles the edge case where OpenAI returns a null body.
    message = response.choices[0].message.content
    return (message or "").strip()


# ---------------------------------------------------------------------------
# Utility: retrieve all stored transactions (used by GET /transactions)
# ---------------------------------------------------------------------------

def get_all_transactions(limit: int = 500) -> list[dict]:
    """
    Fetch up to `limit` raw transaction records from ChromaDB.

    Unlike semantic_search, this function does NOT perform any vector similarity
    search \u2014 it simply returns records in their stored order.  It is used by
    the GET /transactions endpoint to let users browse all ingested data.

    Parameters
    ----------
    limit : int, optional
        Maximum number of records to return.  Defaults to 500.
        The API layer further constrains this to the range [1, 1000].

    Returns
    -------
    list[dict]
        A list of dictionaries, each representing one stored transaction:
          - "id"       : The unique record ID (e.g. "statement.pdf_0").
          - "document" : The raw pipe-delimited transaction text string.
          - "metadata" : Dict with keys "source", "chunk_index", "total_chunks".

    Raises
    ------
    QueryError
        If ChromaDB cannot be accessed.
    """

    try:
        collection = get_collection()
        # collection.get() retrieves records by ID or, without filters, returns
        # up to `limit` records from the collection in storage order.
        results = collection.get(limit=limit)
    except Exception as exc:
        logger.exception("Failed to fetch stored transactions")
        raise QueryError("Could not fetch transactions") from exc

    # ChromaDB returns three parallel lists of equal length.
    # We zip them together into a list of per-record dictionaries.
    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    # Use index-based iteration because zip() on three lazy sequences could
    # silently truncate if any list is shorter than the others.
    return [
        {
            "id": ids[index],
            "document": documents[index],
            "metadata": metadatas[index],
        }
        for index in range(len(ids))
    ]