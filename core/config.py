import os

# folder where PDFs are stored
PDF_FOLDER = "uploads"

# where chromadb stores vectors
CHROMA_PATH = "database"

# collection name used for transaction embeddings
COLLECTION_NAME = "transactions"

# embedding model
EMBED_MODEL = "all-MiniLM-L6-v2"

# OpenAI model for response generation
OPENAI_MODEL = "gpt-4o-mini"

# API guard rails
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_TOP_K = 5

# OpenAI key - set via environment variable: $env:OPENAI_API_KEY="sk-..."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")