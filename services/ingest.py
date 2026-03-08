import logging

from services.parser import parse_pdf
from services.store import get_collection, get_embedding_model


logger = logging.getLogger(__name__)


class IngestionError(Exception):
    pass


def ingest_file(filepath: str, doc_id: str) -> dict:
    try:
        transactions = parse_pdf(filepath)
    except Exception as exc:
        logger.exception("Failed to parse PDF: %s", filepath)
        raise IngestionError("Could not parse PDF file") from exc

    if not transactions:
        raise IngestionError("No transactions found in the uploaded PDF")

    try:
        collection = get_collection()
    except Exception as exc:
        logger.exception("Failed to initialize collection")
        raise IngestionError("Could not initialize vector database") from exc
    existing = collection.get(where={"source": doc_id})
    if existing.get("ids"):
        logger.info("Skipping duplicate document: %s", doc_id)
        return {
            "status": "skipped",
            "reason": "Document already ingested",
            "transactions_added": 0,
        }

    try:
        embedding_model = get_embedding_model()
        embeddings = embedding_model.encode(transactions).tolist()
    except Exception as exc:
        logger.exception("Failed to create embeddings for %s", doc_id)
        raise IngestionError("Could not generate embeddings") from exc

    ids = [f"{doc_id}_{index}" for index in range(len(transactions))]

    metadatas = [
        {
            "source": doc_id,
            "chunk_index": index,
            "total_chunks": len(transactions),
        }
        for index in range(len(transactions))
    ]

    try:
        collection.add(
            documents=transactions,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
    except Exception as exc:
        logger.exception("Failed to store transactions for %s", doc_id)
        raise IngestionError("Could not store transactions") from exc

    logger.info("Ingested %s transactions from %s", len(transactions), doc_id)
    return {
        "status": "ingested",
        "reason": None,
        "transactions_added": len(transactions),
    }