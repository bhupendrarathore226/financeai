from sentence_transformers import SentenceTransformer
import chromadb
import config

model = SentenceTransformer(config.EMBED_MODEL)

client = chromadb.PersistentClient(path=config.CHROMA_PATH)

collection = client.get_collection("transactions")

def semantic_search(question, top_k=5):

    q_embed = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=q_embed,
        n_results=top_k
    )

    return results["documents"][0]

from openai import OpenAI

client_openai = OpenAI()

def ask_llm(question):

    context = semantic_search(question)

    prompt = f"""
Answer the question using these transactions:

{context}

Question: {question}
"""

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content