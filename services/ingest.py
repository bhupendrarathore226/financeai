"""
Ingestion service for FinanceAI.

Responsibility
--------------
This module orchestrates the full pipeline that takes a raw PDF file on disk
and stores its transactions as searchable vector embeddings in ChromaDB.

Pipeline stages
---------------
  1. **Parse**   — Extract transaction rows from the PDF (services/parser.py).
  2. **Deduplicate** — Check if this document was already ingested; skip if so.
  3. **Embed**   — Convert each transaction string to a numeric vector
                   using the SentenceTransformer model (services/store.py).
  4. **Store**   — Persist documents, embeddings, IDs, and metadata in ChromaDB.

Design decisions
----------------
- `IngestionError` is a custom exception that wraps all expected failure modes
  (parse failure, empty document, embedding failure, storage failure).  The
  API layer catches this specific type and maps it to an HTTP 400 response,
  signalling a problem with the *input* rather than a server fault.
- Deduplication is done by querying ChromaDB for any existing document whose
  `source` metadata field matches the uploaded filename.  This prevents the
  same bank statement from being stored twice if re-uploaded.
"""

import logging  # Standard library: structured log messages

# Service dependencies — parse_pdf reads the PDF; get_collection and
# get_embedding_model access ChromaDB and the SentenceTransformer model.
from services.parser import parse_pdf
from services.store import get_collection, get_embedding_model


# Module-level logger.  Log records will appear as "services.ingest" in output.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception type
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """
    Raised when a known, recoverable error prevents a file from being ingested.

    Examples of conditions that raise this exception:
      - The PDF cannot be opened or parsed.
      - The PDF contains no transaction rows (e.g. a scanned image).
      - The vector database cannot be initialised.
      - Embedding generation fails for the extracted text.
      - Storing the embeddings in ChromaDB fails.

    The API layer (api/main.py) catches `IngestionError` specifically and
    returns an HTTP 400 status code with the error message, distinguishing
    it from unexpected server errors (HTTP 500).
    """
    pass  # No additional behaviour needed; the class name and message are enough.


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def ingest_file(filepath: str, doc_id: str) -> dict:
    """
    Parse, embed, and store all transactions from a PDF bank statement.

    Parameters
    ----------
    filepath : str
        Absolute path to the PDF file on disk.
        This file must already exist at the time of the call.
    doc_id : str
        A unique string identifier for this document — typically the
        sanitised filename (e.g. "statement_march_2026.pdf").
        Used as the `source` metadata field and as a prefix for record IDs,
        so it must be stable and unique across all uploaded files.

    Returns
    -------
    dict
        A summary dictionary with keys:
          - "status"             : "ingested" if new rows were added,
                                   "skipped" if the document was already stored.
          - "reason"             : Human-readable explanation (None for "ingested").
          - "transactions_added" : Number of rows written to ChromaDB (0 if skipped).

    Raises
    ------
    IngestionError
        For any expected failure at any stage of the pipeline (see class docstring).
    """

    # ------------------------------------------------------------------
    # Stage 1: Parse the PDF into a list of transaction strings
    # ------------------------------------------------------------------
    try:
        transactions = parse_pdf(filepath)
    except Exception as exc:
        # Log the full traceback so developers can diagnose parsing errors,
        # then re-raise as IngestionError so the API returns a clean 400.
        logger.exception("Failed to parse PDF: %s", filepath)
        raise IngestionError("Could not parse PDF file") from exc

    # If parsing succeeded but produced no rows, the PDF is likely a scanned
    # image or an empty document — reject it with a descriptive message.
    if not transactions:
        raise IngestionError("No transactions found in the uploaded PDF")

    # ------------------------------------------------------------------
    # Stage 2: Initialise the vector database collection
    # ------------------------------------------------------------------
    try:
        collection = get_collection()
    except Exception as exc:
        logger.exception("Failed to initialize collection")
        raise IngestionError("Could not initialize vector database") from exc

    # ------------------------------------------------------------------
    # Stage 3: Deduplication check
    # ------------------------------------------------------------------
    # Query ChromaDB for any existing records tagged with this doc_id as
    # their `source`.  If any IDs are returned, the document is already stored.
    # `where` is a ChromaDB metadata filter — it returns only records whose
    # `source` metadata field equals doc_id.
    existing = collection.get(where={"source": doc_id})
    if existing.get("ids"):
        logger.info("Skipping duplicate document: %s", doc_id)
        return {
            "status": "skipped",
            "reason": "Document already ingested",
            "transactions_added": 0,
        }

    # ------------------------------------------------------------------
    # Stage 4: Generate embeddings for each transaction
    # ------------------------------------------------------------------
    # embedding_model.encode() converts a list of strings into a 2-D NumPy
    # array of shape (len(transactions), embedding_dimension).
    # .tolist() converts the NumPy array to a plain Python list of lists,
    # which is the format ChromaDB expects.
    try:
        embedding_model = get_embedding_model()
        embeddings = embedding_model.encode(transactions).tolist()
    except Exception as exc:
        logger.exception("Failed to create embeddings for %s", doc_id)
        raise IngestionError("Could not generate embeddings") from exc

    # ------------------------------------------------------------------
    # Stage 5: Build IDs and metadata for each record
    # ------------------------------------------------------------------
    # ChromaDB requires a unique string ID for every record.  We build it
    # as "<doc_id>_<index>" (e.g. "statement.pdf_0", "statement.pdf_1") so
    # the source document can always be inferred from the ID.
    ids = [f"{doc_id}_{index}" for index in range(len(transactions))]

    # Metadata is stored alongside each document.  It can be used later for
    # filtering queries (e.g. "only search transactions from file X").
    # - source       : Which file this transaction came from.
    # - chunk_index  : Position of this row within the document (0-based).
    # - total_chunks : Total rows in the document (useful for reconstruction).
    metadatas = [
        {
            "source": doc_id,
            "chunk_index": index,
            "total_chunks": len(transactions),
        }
        for index in range(len(transactions))
    ]

    # ------------------------------------------------------------------
    # Stage 6: Persist to ChromaDB
    # ------------------------------------------------------------------
    # collection.add() writes all four parallel arrays (documents, embeddings,
    # IDs, metadata) to the vector store in a single atomic operation.
    # If any item in the batch fails, ChromaDB rolls back the whole add.
    try:
        collection.add(
            documents=transactions,   # Raw text strings
            embeddings=embeddings,    # Corresponding numeric vectors
            ids=ids,                  # Unique record identifiers
            metadatas=metadatas,      # Source file and positional info
        )
    except Exception as exc:
        logger.exception("Failed to store transactions for %s", doc_id)
        raise IngestionError("Could not store transactions") from exc

    logger.info("Ingested %s transactions from %s", len(transactions), doc_id)

    # Return a success summary for the API layer to pass back to the client.
    return {
        "status": "ingested",
        "reason": None,                          # No warning or skip reason
        "transactions_added": len(transactions), # How many rows were stored
    }