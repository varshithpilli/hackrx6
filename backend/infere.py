from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
import numpy as np

def get_chunks(query: str, chunks, embeddings, top_k: int = 5):

    query_embedding = embedder.encode([query])
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_docs = []
    for idx in top_indices:
        if similarities[idx] > 0.1:
            relevant_docs.append({
                'content': chunks[idx]
            })
    return relevant_docs