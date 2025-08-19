from abc import ABC, abstractmethod
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        pass

class SBERTEmbedder(Embedder):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, convert_to_numpy=True))

class VectorStore(ABC):
    @abstractmethod
    def add(self, embeddings: np.ndarray, documents: List[str]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        pass

class FAISSVectorStore(VectorStore):
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []

    def add(self, embeddings: np.ndarray, documents: List[str]) -> None:
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]
