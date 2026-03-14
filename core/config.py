"""
Application-wide configuration for FinanceAI.

Design principle
----------------
All hard-coded values and environment variables are centralised here.
Every other module imports from this file rather than sprinkling magic
strings or numbers throughout the codebase.  This makes it trivial to
change a value in one place and have the effect propagate everywhere.

Environment variables
---------------------
Sensitive credentials (e.g. the OpenAI API key) are NEVER hard-coded.
Instead they are read from the process environment at startup so that:
  1. Secrets are not accidentally committed to version control.
  2. Different environments (dev / staging / prod) can use different keys
     without changing a single line of source code.

How to set the OpenAI API key (Windows PowerShell example):
    $env:OPENAI_API_KEY = "sk-..."
"""

import os  # Standard library module for interacting with the operating system

# ---------------------------------------------------------------------------
# File-system paths
# ---------------------------------------------------------------------------

# Directory where uploaded PDF bank statements are saved before ingestion.
# The ingestion service reads files from this folder.
PDF_FOLDER = "uploads"

# Directory used by ChromaDB to persist its vector store on disk.
# ChromaDB is an embedded vector database — storing data here means it
# survives server restarts without needing an external database service.
CHROMA_PATH = "database"

# ---------------------------------------------------------------------------
# Vector database settings
# ---------------------------------------------------------------------------

# Logical name for the ChromaDB collection that holds transaction embeddings.
# A "collection" in ChromaDB is analogous to a table in a relational DB.
COLLECTION_NAME = "transactions"

# Name of the sentence-transformer model used to convert text into vectors.
# "all-MiniLM-L6-v2" is a lightweight but accurate model that runs locally
# on CPU, so no internet connection is needed after the initial download.
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# OpenAI / LLM settings
# ---------------------------------------------------------------------------

# The OpenAI chat model used to generate natural-language answers.
# "gpt-4o-mini" balances cost and quality well for finance Q&A tasks.
OPENAI_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# API safety limits
# ---------------------------------------------------------------------------

# Maximum permitted file upload size (10 MB expressed in bytes).
# Files larger than this are rejected before they touch the disk, preventing
# denial-of-service via enormous uploads.
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Default number of semantically similar transaction chunks returned per
# vector search.  Increasing this gives the LLM more context but also
# increases prompt size and therefore costs more tokens.
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# Secrets — loaded from environment variables at startup
# ---------------------------------------------------------------------------

# os.getenv() returns None if the variable is not set, which will cause
# OpenAI calls to fail with a clear authentication error rather than crashing
# at import time.  See the docstring above for how to set this variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")