import os

# folder where PDFs are stored
PDF_FOLDER = "uploads"

# where chromadb stores vectors
CHROMA_PATH = "database"

# embedding model
EMBED_MODEL = "all-MiniLM-L6-v2"

# OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")