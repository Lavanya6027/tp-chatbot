import os
from extractors import ExtractorDispatcher, PDFExtractor, DocxExtractor, TxtExtractor
from preprocessing import Normalizer, Chunker
from embeddings import SBERTEmbedder, FAISSVectorStore
from rag import RAGPipeline, Chatbot

# Register extractors
ExtractorDispatcher.register(".pdf", PDFExtractor())
ExtractorDispatcher.register(".docx", DocxExtractor())
ExtractorDispatcher.register(".txt", TxtExtractor())

# Initialize components
normalizer = Normalizer()
chunker = Chunker()
embedder = SBERTEmbedder()
store = FAISSVectorStore(embedding_dim=384)
rag = RAGPipeline(embedder, store, chunker, normalizer)
bot = Chatbot(rag)

def clear_console():
    # Windows
    if os.name == 'nt':
        os.system('cls')
    # Linux / Mac
    else:
        os.system('clear')

# Ingest documents
docs_folder = r"C:\Users\shenile.a\Downloads\my_docs"
for f in os.listdir(docs_folder):
    path = os.path.join(docs_folder, f)
    try:
        text = ExtractorDispatcher.extract(path)
        rag.add_document(text)
        print(f"Ingested {f}")
    except Exception as e:
        print(f"Skipped {f}: {e}")
        
clear_console()

# Chat loop
while True:
    q = input("\nYou: ")
    if q.lower() in ["exit", "quit"]:
        break
    # For now, no keyword filtering
    print("\nBot:\n", bot.answer(q, top_k=5, similarity_threshold=0.0, keywords=None))

