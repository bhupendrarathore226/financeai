# FinanceAI: Technical Interview Presentation
## Senior Software Engineer Level

---

## Opening Statement (The Pitch)

FinanceAI is a production-oriented Retrieval-Augmented Generation (RAG) API that allows
users to query their personal finance data using natural language. Instead of writing SQL
or filtering spreadsheets, a user can simply ask "How much did I spend on dining in
February?" and receive a contextually accurate answer grounded in their actual bank
statement data.

The core engineering challenge was building a pipeline that bridges the gap between raw
PDF documents and semantic, LLM-powered question answering - while keeping the
architecture simple enough to evolve and maintain.

---

## 1. Architecture Explanation

### Style: Layered Modular Monolith with RAG Pipeline

The system is deliberately structured as a layered monolith rather than microservices.
There are four clear horizontal layers:

    Layer 1 - Transport (api/)
      FastAPI endpoints: /upload, /chat, /transactions
      Responsibility: HTTP contract, input validation, error mapping

    Layer 2 - Service (services/)
      parser.py   -> PDF extraction
      ingest.py   -> orchestration: parse + embed + store
      query.py    -> semantic search + LLM answer generation
      Responsibility: all business logic lives here

    Layer 3 - Infrastructure (services/store.py)
      Lazy-initialized wrappers for ChromaDB and SentenceTransformer
      Responsibility: external client lifecycle management

    Layer 4 - Configuration (core/config.py)
      All runtime constants, model names, paths, env-var bindings
      Responsibility: single source of truth for settings

### RAG Pipeline

The RAG pattern is the backbone of the chat feature:

    Ingest time:
    PDF -> pdfplumber -> text rows -> SentenceTransformer embeddings -> ChromaDB

    Query time:
    Question -> SentenceTransformer embedding -> ChromaDB top-k retrieval
             -> prompt assembly -> GPT-4o-mini -> answer

This two-phase approach is intentional: ingestion is expensive and one-time, while
retrieval is cheap and latency-sensitive.

### Technology Choices Summary

    Component          Technology              Why
    ───────────────────────────────────────────────────────────────
    API framework      FastAPI                 async-ready, built-in Pydantic validation
    PDF parsing        pdfplumber              best table extraction from financial PDFs
    Embeddings         sentence-transformers   fast, local, no API cost for embeddings
    Vector store       ChromaDB                zero-infra local persistence, SQL-backed
    LLM                OpenAI GPT-4o-mini      cost/quality balance for short-context Q&A
    Validation         Pydantic (via FastAPI)  declarative, integrated, type-safe

---

## 2. Key Design Decisions

### Decision 1: Lazy Singleton Initialization

All heavy clients (ChromaDB, SentenceTransformer, OpenAI) are initialized once on first
use, not at module import time.

    Why: Import-time initialization causes crashes during testing when dependencies are
    unavailable, slows down startup, and makes mocking painful.

    How: Module-level private variables (_client, _embedding_model, _openai_client)
    guarded by None checks inside factory functions (get_collection,
    get_embedding_model, get_openai_client).

    Tradeoff: First real request is slower; subsequent requests pay zero init cost.

### Decision 2: Service Layer Isolation

Business rules live exclusively in services/. The API layer only handles HTTP concerns.

    Why: If you swap FastAPI for gRPC or a CLI, the business logic is untouched.
    If you want to unit-test ingest logic, you do not need an HTTP server.

    How: api/main.py calls ingest_file(), ask_llm(), get_all_transactions() with
    no business logic of its own.

### Decision 3: Domain Error Types

Two custom exceptions - IngestionError and QueryError - are defined.

    Why: Without them, all failures bubble up as generic Exception or HTTP 500.
    Service code raises typed errors which the API layer catches and maps to the
    appropriate HTTP status code (400 vs 500).

    How: Service functions catch low-level SDK exceptions and re-raise as
    IngestionError/QueryError with safe, user-facing messages.

### Decision 4: Deterministic Duplicate Prevention

Before ingesting a file, the system checks ChromaDB for existing documents with
{source: doc_id} in metadata.

    Why: Without this, re-uploading the same statement doubles every transaction in
    the vector store, corrupting spending totals and retrieval quality.

    How: collection.get(where={"source": doc_id}) --- if any IDs exist, skip
    and return status="skipped".

### Decision 5: Filename Sanitization Against Path Traversal

Uploaded filenames are sanitized using Path(file.filename).name.

    Why: An attacker could upload a file named ../../etc/passwd and cause the
    application to write outside the uploads directory.

    How: .name extracts only the final component, stripping any directory separators.

### Decision 6: Local Embeddings, Remote LLM

Embeddings are generated locally using sentence-transformers (all-MiniLM-L6-v2).
Final answer generation uses the OpenAI API.

    Why: Embedding every transaction row via OpenAI's API would be expensive and
    slow at ingestion time. Local embeddings are fast, free, and consistent.
    LLM generation is reserved for the final step where quality matters most.

---

## 3. Trade-offs

### Trade-off 1: Monolith vs. Microservices

    Chosen: Monolith
    Pro: Simpler deployment, no network serialization overhead, easier local dev,
         no service discovery needed.
    Con: Ingestion and query share the same process; a long ingestion job blocks
         server capacity. Cannot scale ingestion and API independently.
    Mitigation: Add background task queues (FastAPI BackgroundTasks / Celery) to
                decouple ingestion from the request/response cycle.

### Trade-off 2: Local Vector Store vs. Managed Service

    Chosen: ChromaDB with local disk persistence
    Pro: Zero infra cost, no network latency, works offline.
    Con: Not distributed; cannot be shared across multiple server instances;
         no built-in backup, HA, or replication.
    Mitigation: Swap to Pinecone, Weaviate or Qdrant for production. The store.py
                wrapper layer means only one file needs to change.

### Trade-off 3: Full File Load vs. Streaming Upload

    Chosen: Read full file into memory before writing
    Pro: Simple; allows size validation before touching disk.
    Con: For the 10 MB limit, this is fine. If the limit grows, peak memory per
         concurrent upload becomes a concern.
    Mitigation: Stream to disk, then validate file size from on-disk bytes.

### Trade-off 4: Local Embedding Model vs. API Embeddings

    Chosen: Local sentence-transformers
    Pro: No per-token cost, no external dependency at ingest time, deterministic.
    Con: First load downloads ~90 MB model; slightly lower quality than
         text-embedding-3-large.
    Mitigation: Make EMBED_MODEL configurable in config.py and support an
                OpenAI embedding backend behind the same interface.

### Trade-off 5: No Authentication

    Chosen: Open API (no auth)
    Pro: Simpler for a personal local tool.
    Con: Not safe to expose on any network. Anyone can upload files, read all
         transactions, or exhaust OpenAI API quota.
    Mitigation: Add API key header auth or OAuth2 before any external deployment.

### Trade-off 6: CORS Wildcard

    Chosen: allow_origins=["*"] with allow_credentials=True
    Pro: Convenient for local frontend development.
    Con: Dangerous in production - allows any origin to make credentialed requests.
    Mitigation: Restrict to explicit trusted domains in production config.

---

## 4. Scalability Considerations

### Vertical Scaling (Scale Up)

The current design scales vertically with minimal changes:
- Uvicorn supports multiple workers (--workers 4).
- The embedding model and Chroma client are process-local singletons; each worker
  carries its own initialized instance after first use.
- ChromaDB's SQLite backend has write serialization limits; this is the first
  bottleneck under high ingest volume.

### Horizontal Scaling (Scale Out)

Horizontal scaling requires three targeted changes:

    1. Replace ChromaDB local with a network-accessible vector DB
       (Qdrant / Weaviate / Pinecone with consistent connection string in config).

    2. Externalize embedding model to a dedicated inference service
       (TorchServe / Triton / SageMaker) so all API pods share one model server.

    3. Decouple ingestion with a task queue
       Upload endpoint writes PDF to object storage (S3/GCS) and enqueues a job.
       A separate worker pool runs ingest_file() asynchronously.
       This decouples upload latency from ingestion throughput entirely.

### Throughput Bottlenecks (in order)

    Bottleneck 1: OpenAI API rate limits (chat throughput is bounded by tokens/min).
    Bottleneck 2: SentenceTransformer inference (CPU-bound per request at query time).
    Bottleneck 3: ChromaDB SQLite write lock (multiple concurrent ingest workers).
    Bottleneck 4: PDF parsing (pdfplumber is synchronous and CPU-intensive).

### Caching Opportunities

    - Repeated identical questions: cache (question_hash -> answer) in Redis/Memcached
      with a configurable TTL.
    - Embedding of frequently uploaded documents: memoize by content hash.
    - Collection metadata (total count): cache to avoid repeated Chroma calls.

---

## 5. Challenges Solved in the Project

### Challenge 1: Turning Unstructured PDFs into Queryable Data

Bank statement PDFs have no standard schema - some use tables, some use columns, some
mix both. pdfplumber's extract_tables() handles tabular layouts. The _clean_row()
function normalizes each row into a stable pipe-delimited string, making embeddings
consistent regardless of source formatting quirks.

### Challenge 2: Preventing Vector Store Poisoning via Duplicate Ingestion

Without deduplication, re-uploading the same statement makes every transaction appear
twice. The solution is metadata-scoped lookup: before any write, the system checks for
existing records with matching source. This is a simple but robust guard that works
purely with Chroma's native query API.

### Challenge 3: Keeping Tests Fast and Hermetic Without Real Infrastructure

The test suite is fully isolated from real ChromaDB and OpenAI. This is achieved by:
- Stubbing services.store module-level with a lightweight types.ModuleType fake before
  import (patching at the sys.modules level, not per-test).
- Mocking ingest_file, ask_llm, get_all_transactions at the API boundary.
This means tests run in milliseconds, require no environment variables, and never
make network calls.

### Challenge 4: Separating Transport Concerns from Business Errors

FastAPI raises HTTPException; services raise domain errors. The API layer bridges them:
IngestionError -> 400, QueryError -> 400, unexpected Exception -> 500. This means the
service layer is HTTP-agnostic - you can call ingest_file() from a CLI, a test, or a
background job with identical behavior.

### Challenge 5: Controlling Prompt Quality for Finance Domain

RAG quality depends entirely on what retrieval returns. Two decisions keep prompts
grounded:
1. Context is limited to top-k semantically similar transaction rows, not entire history.
2. The system prompt explicitly instructs the model to say "I am not sure" if the
   answer cannot be inferred from the provided context, reducing hallucination.

---

## Interview Question Preparation

Q: Why RAG instead of fine-tuning a model on transaction data?
A: Fine-tuning bakes knowledge into model weights at training time - it cannot adapt
to new data without retraining. RAG retrieves fresh data at inference time, which is
exactly what we need: every new statement upload is immediately queryable without any
model update.

Q: How would you make the query results more accurate?
A: (a) Use a re-ranker after initial retrieval to reorder chunks by relevance.
   (b) Add date/amount metadata filters to pre-scope retrieval before vector search.
   (c) Add query expansion - generate alternative phrasings of the user's question
       to improve recall from the vector store.
   (d) Increase chunk granularity: embed individual fields (date, merchant, amount)
       separately for more precise matching.

Q: What would you change first if this went to production?
A: Authentication immediately - the API currently exposes all financial data openly.
Second, move ingestion to a background task queue so the upload endpoint returns fast.
Third, replace CORs wildcard with an explicit allowlist.

Q: What is the biggest architectural risk right now?
A: The ChromaDB SQLite backend is a single write-serialized file on local disk. Any
process restart loses nothing (it persists), but you cannot run multiple API instances
sharing the same store, and there is no backup strategy. The risk is data loss on disk
failure and inability to scale out.

Q: How does this handle security of financial data?
A: Currently it does not - there is no auth, data is stored in plaintext on local disk,
and the API is fully open. For a real deployment: encrypt the Chroma data at rest,
add auth with per-user collection namespacing so users can only retrieve their own data,
and audit-log all uploads and queries.

---

## Summary

FinanceAI demonstrates a sound foundation for a RAG-based document Q&A system:
clean layering, testable services, defensive input validation, and a clear separation
between the ingestion pipeline and the query pipeline.

The key architectural decisions - lazy initialization, typed domain errors, deduplication
by metadata, local embeddings - are each justified tradeoffs rather than accidental
choices. The system is intentionally kept simple for a personal-finance use case, but
the upgrade path to production is clear and each component has an identified migration
target.
