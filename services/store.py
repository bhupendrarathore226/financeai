from typing import Any

import core.config as config


# Lazy singletons avoid import-time crashes and speed startup.
_embedding_model = None
_client = None


def _get_client() -> Any:
    global _client
    if _client is None:
        try:
            import chromadb
        except Exception as exc:
            raise RuntimeError("ChromaDB is not available in this environment") from exc

        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return _client


def get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("SentenceTransformer is not available in this environment") from exc

        _embedding_model = SentenceTransformer(config.EMBED_MODEL)
    return _embedding_model


def get_collection():
    client = _get_client()
    return _client.get_or_create_collection(name=config.COLLECTION_NAME)
