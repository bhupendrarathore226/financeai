"""
FastAPI application — HTTP route definitions for FinanceAI.

Architecture overview
---------------------
This module is the **HTTP layer** of the application.  Its only job is to:
  1. Accept and validate incoming HTTP requests.
  2. Delegate actual work to the service layer (services/).
  3. Map service-level errors to appropriate HTTP status codes.
  4. Return well-structured JSON responses.

It intentionally contains NO business logic.  All PDF parsing, embedding
generation, vector storage, and LLM calls live in the `services/` package.

Endpoints exposed
-----------------
  POST /upload       — Upload a PDF bank statement for ingestion.
  POST /chat         — Ask a natural-language question about transactions.
  GET  /transactions — Retrieve stored transactions (paginated by `limit`).

Dependency chain (high level)
------------------------------
  HTTP request
      └── api/main.py  (this file — validates input, calls services)
              ├── services/ingest.py  (PDF → embeddings → ChromaDB)
              └── services/query.py   (question → embeddings → ChromaDB → OpenAI → answer)
"""

import logging          # Standard library: structured log messages
from pathlib import Path  # Standard library: OS-agnostic file path manipulation

# FastAPI is the web framework.  Each symbol imported here has a specific role:
#   FastAPI     — The application class that wires everything together.
#   File        — Declares that a route parameter comes from a multipart file upload.
#   HTTPException — Raise this to send a non-200 HTTP response with a JSON error body.
#   UploadFile  — Represents the uploaded file object (exposes .filename, .read(), etc.).
from fastapi import FastAPI, File, HTTPException, UploadFile

# Pydantic is used for request body validation.
#   BaseModel — Parent class for all request/response schemas.
#   Field     — Adds validation constraints (min/max length, regex, etc.) to fields.
from pydantic import BaseModel, Field

# Allows browsers running on different origins (e.g. a React frontend on
# localhost:3000) to call this API (which may run on localhost:8000).
# Without this middleware, the browser's same-origin policy would block the requests.
from fastapi.middleware.cors import CORSMiddleware

import core.config as config  # Centralised configuration (paths, limits, model names)

# Import service functions and their custom error types.
# Importing error types explicitly lets us catch them precisely and map them
# to meaningful HTTP status codes instead of a generic 500.
from services.ingest import IngestionError, ingest_file
from services.query import QueryError, ask_llm, get_all_transactions


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Configure the root logger once at module level.  All loggers created with
# logging.getLogger(__name__) throughout the app will inherit this format.
# Format breakdown:
#   %(asctime)s   — human-readable timestamp
#   %(levelname)s — DEBUG / INFO / WARNING / ERROR / CRITICAL
#   [%(name)s]    — logger name (usually the module path, e.g. api.main)
#   %(message)s   — the actual log message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Module-level logger — use `logger.info(...)`, `logger.exception(...)`, etc.
# throughout this file instead of print() so messages are consistently formatted.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------
# `app` is the central FastAPI object.  Route decorators (@app.post, @app.get)
# register handlers on this instance.  The ASGI server (uvicorn) receives an
# HTTP request, finds the matching handler, and calls it.
app = FastAPI()

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
# Cross-Origin Resource Sharing (CORS) is a browser security mechanism that
# blocks JavaScript on one origin from calling an API on a different origin.
# Adding this middleware tells the browser that all origins are allowed,
# which is fine for development / internal tools.
#
# SECURITY NOTE FOR PRODUCTION: Replace allow_origins=["*"] with a specific
# list like allow_origins=["https://your-frontend-domain.com"] to prevent
# untrusted websites from calling this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Accept requests from any domain
    allow_credentials=True,     # Allow cookies / auth headers
    allow_methods=["*"],        # Accept GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],        # Accept any request header
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Pydantic schema for the /chat request body.

    FastAPI automatically deserialises the incoming JSON body into this object
    and validates the constraints defined by `Field` before the handler runs.
    If validation fails, FastAPI returns a 422 Unprocessable Entity response
    automatically — no manual validation code needed.

    Attributes
    ----------
    question : str
        The natural-language question the user wants answered.
        Must be at least 3 characters (to block trivially empty questions)
        and at most 1000 characters (to cap LLM token usage).
    """

    question: str = Field(min_length=3, max_length=1000)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF bank statement and ingest its transactions into ChromaDB.

    This endpoint is the entry point for adding new financial data to the
    system.  The workflow is:
        1. Validate the file (name present, correct content-type, size limit).
        2. Save the file to the upload directory on disk.
        3. Call the ingestion service to parse and embed the transactions.
        4. Return a summary of what was ingested.

    Parameters
    ----------
    file : UploadFile
        The multipart-encoded PDF file sent by the client.
        FastAPI automatically binds the uploaded file to this parameter.

    Returns
    -------
    dict
        JSON response with keys:
          - status           : "ingested" or "skipped" (duplicate file)
          - filename         : The sanitised filename stored on disk
          - transactions_added : Number of transaction rows successfully ingested
          - reason           : Human-readable note (e.g. why a file was skipped)

    Raises
    ------
    HTTPException 400
        If the file has no name, is not a PDF, or the ingestion service
        returns a business-logic error (e.g. no transactions found).
    HTTPException 413
        If the uploaded file exceeds MAX_UPLOAD_SIZE_BYTES.
    HTTPException 500
        If an unexpected internal error occurs during ingestion.
    """

    # --- Step 1: Basic input validation ---

    # Reject requests that have no filename attached; this prevents processing
    # anonymous uploads that could not be tracked or deduplicated.
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    # Only accept PDF MIME types.  Accepting arbitrary files could allow
    # malicious content to be stored on the server.
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # --- Step 2: Sanitise the filename (security: prevent path traversal) ---
    # Path traversal attacks occur when a user provides a filename like
    # "../../etc/passwd".  Path(...).name strips all directory components,
    # leaving only the bare filename (e.g. "statement.pdf").
    safe_name = Path(file.filename).name

    # Ensure the upload directory exists; mkdir(parents=True) creates any
    # missing parent directories; exist_ok=True avoids an error if it already exists.
    upload_dir = Path(config.PDF_FOLDER)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_name  # Full path where the file will be saved

    # --- Step 3: Read the file content and validate size ---
    # Read the entire file into memory ONCE so we can check the size before
    # writing to disk.  This avoids storing oversized files even temporarily.
    contents = await file.read()
    if len(contents) > config.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    # Write the validated content to disk in binary mode ("wb").
    # `with` ensures the file handle is closed even if an error occurs.
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    # --- Step 4: Ingest the file into the vector database ---
    try:
        # ingest_file parses the PDF, creates embeddings, and stores them in
        # ChromaDB.  It returns a summary dict with status and counts.
        result = ingest_file(str(file_path), safe_name)
    except IngestionError as exc:
        # Known business-logic errors (e.g. no transactions found) → 400 Bad Request
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Unexpected errors (e.g. disk full, ChromaDB crash) → 500 Internal Server Error
        # logger.exception logs the full stack trace, which is essential for debugging.
        logger.exception("Unexpected ingestion error for %s", safe_name)
        raise HTTPException(status_code=500, detail="Failed to ingest file") from exc

    logger.info("Upload processed for %s", safe_name)

    # Return only the fields clients need; do not expose raw internal objects.
    return {
        "status": result["status"],
        "filename": safe_name,
        "transactions_added": result["transactions_added"],
        "reason": result["reason"],
    }


@app.post("/chat")
def chat(payload: ChatRequest):
    """
    Answer a natural-language question about the user's financial transactions.

    The workflow is:
        1. Strip leading/trailing whitespace from the question.
        2. Perform a semantic search in ChromaDB to find relevant transactions.
        3. Pass those transactions as context to OpenAI's chat API.
        4. Return the generated answer to the client.

    Parameters
    ----------
    payload : ChatRequest
        The validated request body containing the user's question.
        FastAPI automatically deserialises and validates the JSON body before
        this function is called.

    Returns
    -------
    dict
        JSON response with keys:
          - answer   : The LLM-generated natural-language answer.
          - question : The sanitised question that was asked (echoed for UX).

    Raises
    ------
    HTTPException 400
        If the query service raises a known QueryError (e.g. empty question).
    HTTPException 500
        If an unexpected error occurs during semantic search or LLM generation.
    """

    # Normalise the question: strip() removes accidental leading/trailing spaces
    # or newlines that could affect embedding quality or prompt formatting.
    question = payload.question.strip()

    try:
        # ask_llm orchestrates the full RAG (Retrieval-Augmented Generation) pipeline:
        #   1. Embed the question → 2. Search ChromaDB → 3. Build prompt → 4. Call OpenAI
        answer = ask_llm(question)
    except QueryError as exc:
        # Known errors (e.g. ChromaDB unreachable, empty question) → 400
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Unexpected failures → 500; log the full traceback for debugging
        logger.exception("Unexpected chat error")
        raise HTTPException(status_code=500, detail="Failed to answer question") from exc

    return {"answer": answer, "question": question}


@app.get("/transactions")
def transactions(limit: int = 100):
    """
    Retrieve a paginated list of all stored transaction documents from ChromaDB.

    Useful for debugging, data inspection, or building a transaction history UI.

    Query parameters
    ----------------
    limit : int, optional
        Maximum number of transactions to return.  Defaults to 100.
        Must be between 1 and 1000 (inclusive) to prevent excessively large
        responses that could slow the client or exhaust memory.

    Returns
    -------
    dict
        JSON response with keys:
          - count        : The actual number of transactions returned.
          - transactions : A list of transaction objects, each containing
                           `id`, `document` (raw text), and `metadata`.

    Raises
    ------
    HTTPException 400
        If `limit` is outside the permitted range [1, 1000].
    HTTPException 500
        If ChromaDB cannot be queried due to an internal error.
    """

    # Guard against nonsensical or abusive `limit` values.
    # A limit of 0 would return nothing useful; a limit above 1000 could
    # generate a response large enough to degrade performance.
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")

    try:
        # get_all_transactions fetches raw documents and their metadata from ChromaDB
        items = get_all_transactions(limit=limit)
    except Exception as exc:
        logger.exception("Failed to read transactions")
        raise HTTPException(status_code=500, detail="Failed to fetch transactions") from exc

    # Include the count so the client knows how many items were returned without
    # having to measure the array length on their side.
    return {"count": len(items), "transactions": items}