import os
import logging
from datetime import datetime
from extractors import ExtractorDispatcher, PDFExtractor, DocxExtractor, TxtExtractor
from preprocessing import Normalizer, Chunker
from embeddings import SBERTEmbedder, FAISSVectorStore
from rag import RAGPipeline
from llm import GeminiAIClient, AIAnswerService 
from chatbot import Chatbot
from test_generator import TestGenerator

# Configure logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rag_pipeline_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

logger.info("Starting RAG Pipeline Initialization")

# Register extractors
logger.info("Registering document extractors")
ExtractorDispatcher.register(".pdf", PDFExtractor())
ExtractorDispatcher.register(".docx", DocxExtractor())
ExtractorDispatcher.register(".txt", TxtExtractor())
logger.info("Extractors registered: PDF, DOCX, TXT")

# Initialize components
logger.info("Initializing pipeline components")
try:
    normalizer = Normalizer()
    chunker = Chunker()
    embedder = SBERTEmbedder()
    store = FAISSVectorStore(embedding_dim=384)
    rag = RAGPipeline(embedder, store, chunker, normalizer)
    gemini = GeminiAIClient()
    ai_service = AIAnswerService(gemini)
    tg = TestGenerator(ai_service)
    logger.info("RAG pipeline components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG components: {e}")
    raise

logger.info("Initializing LLM and Chatbot")
try:
    # llm = LocalLLM()
    # cb = Chatbot(rag, llm)
    logger.info("LLM and Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM/Chatbot: {e}")
    raise

def clear_console():
    """Clear console based on OS"""
    # Windows
    if os.name == 'nt':
        os.system('cls')
    # Linux / Mac
    else:
        os.system('clear')

# Ingest documents
docs_folder = r"C:\Users\lavanya.e\Downloads\mydocs"
logger.info(f"Starting document ingestion from: {docs_folder}")

if not os.path.exists(docs_folder):
    logger.error(f"Documents folder not found: {docs_folder}")
    raise FileNotFoundError(f"Folder not found: {docs_folder}")

doc_count = 0
skipped_count = 0

for f in os.listdir(docs_folder):
    path = os.path.join(docs_folder, f)
    try:
        logger.info(f"Processing document: {f}")
        text = ExtractorDispatcher.extract(path)
        q_map = tg.map_questions_to_chunks(text)

        logger.info(f"Generated Q-map for {f}: {len(q_map)} chunks")
        for qid, data in q_map.items():
            print("\n--- Chunk ---")
            print(data["chunk"][:200], "...")
            print("Questions:", data["questions"])
            print("Paraphrased:", data["alt_questions"])
        logger.debug(f"Extracted text length: {len(text)} characters from {f}")
        
        rag.add_document(text)
        doc_count += 1
        logger.info(f"Ingested {f} successfully")
        print(f"Ingested {f}")
        
    except Exception as e:
        skipped_count += 1
        logger.warning(f"Skipped {f}: {e}")
        print(f"Skipped {f}: {e}")

logger.info(f"Document ingestion complete. Success: {doc_count}, Skipped: {skipped_count}")
        
clear_console()
logger.info("Console cleared, starting chat loop")

# Chat loop
logger.info("Entering chat loop")
chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logger.info(f"Chat session started: {chat_session_id}")

while True:
    try:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            logger.info("User requested exit")
            break
        
        logger.info(f"User query: '{q}'")
        
        # Log the retrieval process
        logger.debug("Retrieving relevant chunks from vector store")
        print("\nBot:")

        chunks = rag.retrieve(q, top_k=5, similarity_threshold=0.5)
        logger.info(f"Retrieved {len(chunks)} chunks: {chunks[:2]}")  # show first 2

        response = ai_service.get_answer(q, chunks)

        # For now, no keyword filtering
        # response = cb.answer(q, top_k=5, similarity_threshold=0.0, keywords=None)
        
        # logger.info(f"Generated response length: {len(response)} characters")
        # logger.debug(f"Response content: {response[:200]}...")  # First 200 chars
        
        print(response)
        
    except KeyboardInterrupt:
        logger.warning("Chat interrupted by user (Ctrl+C)")
        break
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        print("Sorry, an error occurred. Please try again.")

logger.info("Chat session ended")
logger.info("RAG Pipeline shutdown complete")