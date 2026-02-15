import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    # 🔹 Chunking function
    def chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    # 🔹 Build index from multiple documents
    def build_index(self, documents):
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)

        self.chunks = all_chunks

        embeddings = self.embedder.encode(self.chunks)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    # 🔹 Retrieve Top-K
    def retrieve(self, query, top_k=3):
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "chunk": self.chunks[idx],
                "score": float(distances[0][i])
            })

        return results
