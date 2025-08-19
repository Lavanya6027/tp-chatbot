from typing import List
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# --- Step 1: Interfaces (SOLID Principle) ---
class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass


class VectorStore(ABC):
    @abstractmethod
    def add(self, embeddings: np.ndarray, documents: List[str]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        pass


# --- Step 2: Concrete Implementations ---
class SBERTEmbedder(Embedder):
    """Sentence-BERT embedder"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, convert_to_numpy=True))


class FAISSVectorStore(VectorStore):
    """FAISS in-memory vector DB"""
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []

    def add(self, embeddings: np.ndarray, documents: List[str]) -> None:
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]


# --- Step 3: RAG Orchestrator ---
class RAGPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def add_documents(self, documents: List[str]) -> None:
        embeddings = self.embedder.embed(documents)
        self.store.add(embeddings, documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedder.embed([query])
        return self.store.search(query_embedding, top_k)


# --- Step 4: Example Usage ---
# if __name__ == "__main__":
#     # Initialize components
#     embedder = SBERTEmbedder()
#     store = FAISSVectorStore(embedding_dim=384)  # SBERT dimension = 384
#     rag = RAGPipeline(embedder, store)

#     # Add documents
#     docs = [
#         "The car is a fast vehicle.",
#         "Cats are small domesticated animals.",
#         "Automobiles and cars mean the same thing.",
#         "Dogs are loyal pets."
#     ]
#     rag.add_documents(docs)

#     # Ask a query
#     query = "Tell me about automobiles"
#     retrieved = rag.retrieve(query)

#     print(f"Query: {query}")
#     print("Retrieved Docs:", retrieved)
