import logging

from openai import OpenAI

import core.config as config
from services.store import get_collection, get_embedding_model


logger = logging.getLogger(__name__)


class QueryError(Exception):
    pass


def semantic_search(question: str, top_k: int = config.DEFAULT_TOP_K) -> list[str]:
    if not question or not question.strip():
        raise QueryError("Question cannot be empty")

    try:
        embedding_model = get_embedding_model()
        q_embed = embedding_model.encode([question]).tolist()
    except Exception as exc:
        logger.exception("Failed to embed question")
        raise QueryError("Could not prepare search query") from exc

    try:
        collection = get_collection()
        results = collection.query(query_embeddings=q_embed, n_results=top_k)
    except Exception as exc:
        logger.exception("Semantic search failed")
        raise QueryError("Could not query vector database") from exc

    documents = results.get("documents", [])
    if not documents or not documents[0]:
        return []

    return documents[0]


_openai_client = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _build_prompt(question: str, context_chunks: list[str]) -> str:
    if context_chunks:
        # Keep context compact and deterministic for stable prompting.
        context_text = "\n".join(f"- {chunk}" for chunk in context_chunks)
    else:
        context_text = "No transaction context found."

    return (
        "Answer the question using these transactions. "
        "If the answer cannot be inferred, say you are not sure.\n\n"
        f"Transactions:\n{context_text}\n\n"
        f"Question: {question}"
    )


def ask_llm(question: str) -> str:
    context_chunks = semantic_search(question)
    prompt = _build_prompt(question, context_chunks)

    try:
        response = get_openai_client().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.exception("OpenAI completion failed")
        raise QueryError("Could not generate answer from language model") from exc

    message = response.choices[0].message.content
    return (message or "").strip()


def get_all_transactions(limit: int = 500) -> list[dict]:
    try:
        collection = get_collection()
        results = collection.get(limit=limit)
    except Exception as exc:
        logger.exception("Failed to fetch stored transactions")
        raise QueryError("Could not fetch transactions") from exc

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    return [
        {
            "id": ids[index],
            "document": documents[index],
            "metadata": metadatas[index],
        }
        for index in range(len(ids))
    ]