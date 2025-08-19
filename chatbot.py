# chatbot.py
from typing import List
from rag import RAGPipeline

class Chatbot:
    """RAG + LLM Chatbot"""
    def __init__(self, rag: RAGPipeline, llm):
        self.rag = rag
        self.llm = llm

    def answer(self, query: str, top_k: int = 5, similarity_threshold: float = 0.5,
               keywords: List[str] = None, max_new_tokens: int = 256) -> str:
        # 1. Retrieve relevant chunks
        chunks = self.rag.retrieve(query, top_k=top_k, similarity_threshold=similarity_threshold, keywords=keywords)
        if not chunks:
            return "❌ Sorry, no relevant information found."

        # 2. Prepare prompt
        context = "\n".join(chunks)
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        # 3. Generate answer using LLM
        return self.llm.generate(prompt, max_new_tokens=max_new_tokens)
