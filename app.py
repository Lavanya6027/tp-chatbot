from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import RAGPipeline, Chatbot
from embeddings import Embedder, VectorStore
from preprocessing import Normalizer, Chunker

# Create components
embedder = Embedder()
store = VectorStore()
chunker = Chunker()
normalizer = Normalizer()

# Create pipeline
pipeline = RAGPipeline(embedder, store, chunker, normalizer)
chatbot = Chatbot(pipeline)

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    if not query.strip():
        return jsonify({"error": "Query is required"}), 400

    answer = chatbot.answer(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
