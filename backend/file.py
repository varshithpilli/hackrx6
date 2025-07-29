import os
import PyPDF2
import re
from typing import List
import httpx
import requests
import tempfile
import os
from sentence_transformers import SentenceTransformer
import numpy as np
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def download_file(url: str) -> str:
    """Downloads a file from a URL and returns the local file path."""
    response = requests.get(url, stream=True)
    response.raise_for_status() 
    
    temp_dir = tempfile.gettempdir()
    file_name = url.split("?")[0].split("/")[-1]
    file_path = os.path.join(temp_dir, file_name)

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return file_path

def embed_text(chunks):
    """Embeds the chunks"""
    embeddings = np.array(embedder.encode(chunks))
    return embeddings

def chunk_text(file_path: str):
    """Chunks PDF."""
    contents = ""

    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                contents += page.extract_text() + "\n\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")

    contents = re.sub(r"\s+", " ", contents.strip())
    contents = re.sub(r"[^\w\s.,!?()-]", "", contents)

    chunk_size = 200
    overlap = 30
    words = contents.split()

    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
            
    return chunks