# financeai

Personal Finance RAG API that ingests PDF statements, stores transaction embeddings in ChromaDB, and answers natural language questions about spending.

## Pipeline

1. Upload PDF statement through `/upload`
2. Parse rows using `pdfplumber`
3. Convert rows to embeddings with `SentenceTransformers`
4. Store transaction chunks and metadata in local ChromaDB
5. Retrieve relevant chunks with semantic search
6. Ask GPT-4o-mini to answer based on retrieved context

## Project Structure

- `api/main.py`: FastAPI endpoints
- `services/parser.py`: PDF table parsing and row cleanup
- `services/ingest.py`: ingestion orchestration, duplicate checks, metadata
- `services/query.py`: semantic search and LLM answer generation
- `services/store.py`: shared embedding model and Chroma collection
- `core/config.py`: constants and runtime settings

## Prerequisites

- Python 3.10+
- OpenAI API key

Set environment variable:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

## Install

```powershell
pip install -r requirements.txt
```

## Run API

```powershell
uvicorn api.main:app --reload
```

## API Examples

### 1) Upload statement

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
	-F "file=@uploads/statement.pdf"
```

### 2) Ask question

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
	-H "Content-Type: application/json" \
	-d '{"question":"How much did I spend on groceries last month?"}'
```

### 3) List stored transactions

```bash
curl "http://127.0.0.1:8000/transactions?limit=100"
```

## Run Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- Duplicate protection is based on source filename metadata.
- Upload endpoint currently supports PDF files only.
- Chroma data is stored locally in `database/`.
