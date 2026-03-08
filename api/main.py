import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import core.config as config
from services.ingest import IngestionError, ingest_file
from services.query import QueryError, ask_llm, get_all_transactions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Keep only the base filename to avoid directory traversal paths.
    safe_name = Path(file.filename).name
    upload_dir = Path(config.PDF_FOLDER)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_name

    # Read once so we can validate size before writing to disk.
    contents = await file.read()
    if len(contents) > config.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    try:
        result = ingest_file(str(file_path), safe_name)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected ingestion error for %s", safe_name)
        raise HTTPException(status_code=500, detail="Failed to ingest file") from exc

    logger.info("Upload processed for %s", safe_name)
    return {
        "status": result["status"],
        "filename": safe_name,
        "transactions_added": result["transactions_added"],
        "reason": result["reason"],
    }


@app.post("/chat")
def chat(payload: ChatRequest):
    question = payload.question.strip()

    try:
        answer = ask_llm(question)
    except QueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected chat error")
        raise HTTPException(status_code=500, detail="Failed to answer question") from exc

    return {"answer": answer, "question": question}


@app.get("/transactions")
def transactions(limit: int = 100):
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")

    try:
        items = get_all_transactions(limit=limit)
    except Exception as exc:
        logger.exception("Failed to read transactions")
        raise HTTPException(status_code=500, detail="Failed to fetch transactions") from exc

    return {"count": len(items), "transactions": items}