import chromadb
from sentence_transformers import SentenceTransformer
from parser import parse_pdf
import config

# load embedding model
model = SentenceTransformer(config.EMBED_MODEL)

# initialize database
client = chromadb.PersistentClient(path=config.CHROMA_PATH)

collection = client.get_or_create_collection(
    name="transactions"
)

def ingest_file(filepath, doc_id):

    transactions = parse_pdf(filepath)

    embeddings = model.encode(transactions).tolist()

    ids = [f"{doc_id}_{i}" for i in range(len(transactions))]

    metadatas = [{"source": doc_id} for _ in transactions]

    collection.add(
        documents=transactions,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )