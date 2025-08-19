from typing import List
import numpy as np
from embeddings import Embedder, VectorStore
from preprocessing import Normalizer, Chunker

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class RAGPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, chunker: Chunker, normalizer: Normalizer):
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.normalizer = normalizer

    def add_document(self, text: str):
        text = self.normalizer.normalize(text)
        chunks = self.chunker.chunk(text)  # Ensure chunker returns ~100-200 word chunks
        embeddings = self.embedder.embed(chunks)
        self.store.add(embeddings, chunks)

    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = 0.5, keywords: List[str] = None) -> List[str]:
        query_embedding = self.embedder.embed([query])[0]
        distances, indices = self.store.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        retrieved = []

        for dist, idx in zip(distances[0], indices[0]):
            sim = 1 / (1 + dist)  # convert FAISS distance to similarity
            if sim < similarity_threshold:
                continue
            chunk_text = self.store.documents[idx]

            # Optional keyword-based trimming
            if keywords:
                sentences = chunk_text.split(". ")
                matched = []
                for i, sent in enumerate(sentences):
                    if any(kw.lower() in sent.lower() for kw in keywords):
                        # Include sentence plus 1 sentence before & after for context
                        start = max(0, i-1)
                        end = min(len(sentences), i+2)
                        matched.append(". ".join(sentences[start:end]))
                if matched:
                    retrieved.extend(matched)
            else:
                retrieved.append(chunk_text)

        # Sort by similarity descending
        return retrieved[:top_k]

class Chatbot:
    def __init__(self, rag: RAGPipeline):
        self.rag = rag

    def answer(self, query: str, top_k: int = 5, similarity_threshold: float = 0.5, keywords: List[str] = None) -> str:
        chunks = self.rag.retrieve(query, top_k=top_k, similarity_threshold=similarity_threshold, keywords=keywords)
        if not chunks:
            return "❌ Sorry, no relevant information found in the documents."
        return "\n".join(chunks)
