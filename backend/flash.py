from fastapi import FastAPI, Request, Depends, Header, HTTPException
from .model_class_1 import Constitutioner
from .models import QueryRequest, QueryResponse
from .file import download_file, chunk_text, embed_text
from .utils import log_time
import time
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

AUTH_KEY = os.getenv("AUTH_KEY")

def verify_key(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]
    if token.strip() != AUTH_KEY.strip():
        raise HTTPException(status_code=403, detail="Invalid Auth token")

app = FastAPI()

engine = Constitutioner()

@app.get("/")
def landing():
    return {"message": "Landed successfully"}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_key)])
async def ask_query(req: QueryRequest):
    
    start_main = time.perf_counter()
    documents = req.documents if isinstance(req.documents, list) else [req.documents]
    print(documents, end="\n\n")

    for doc_url in documents:
        start = time.perf_counter()
        
        file_path = download_file(doc_url)
        
        end = time.perf_counter()
        log_time("download_file", start, end)
        
        print("document downloaded")
        
        start = time.perf_counter()
        
        chunks = chunk_text(file_path=file_path)
        
        end = time.perf_counter()
        log_time("chunk_text", start, end)
        
        print("document chunked")
        
        start = time.perf_counter()
        
        embeddings = embed_text(chunks)
        
        end = time.perf_counter()
        log_time("embed_text", start, end)
        
        print("chunks embedded")
        
        os.remove(file_path)
        print("Temp file removed")
        
    answers = [engine.inference(query=q, chunks=chunks, embeddings=embeddings) for q in req.questions]
    final_answers = await asyncio.gather(*answers)
    
    end_main = time.perf_counter()
    log_time("Total time", start_main, end_main)
    return QueryResponse(answers=final_answers)