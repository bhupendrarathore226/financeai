from fastapi import FastAPI, UploadFile
from ingest import ingest_file
from query import ask_llm
import shutil

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile):

    filepath = f"uploads/{file.filename}"

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_file(filepath, file.filename)

    return {"status": "uploaded"}

@app.post("/chat")
def chat(question: str):

    answer = ask_llm(question)

    return {"answer": answer}