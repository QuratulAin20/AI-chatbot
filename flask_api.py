import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_groq import ChatGroq
import pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "hybrid-search-langchain-pinecone"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Initialize Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize BM25 Encoder
bm25encoder = BM25Encoder().default()

# Load and preprocess text data
def load_and_preprocess_data(filename="scraped_data.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    
    clean_text = " ".join(text.split())  # Remove extra spaces
    
    with open("cleaned_scrape.txt", "w", encoding="utf-8") as file:
        file.write(clean_text)
    
    # Load documents
    loader = TextLoader("cleaned_scrape.txt", encoding="utf-8")
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    return split_docs

split_docs = load_and_preprocess_data()
split_docs_texts = [doc.page_content for doc in split_docs]

# Fit BM25 Encoder on the text
bm25encoder.fit(split_docs_texts)
bm25encoder.dump("bm25_values.json")

# Reload BM25 Encoder
bm25encoder = BM25Encoder().load("bm25_values.json")

# Initialize Hybrid Retriever
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, 
    sparse_encoder=bm25encoder, 
    index=index
)

# Insert documents into Pinecone
retriever.add_texts(split_docs_texts)

# Initialize LLM (GROQ)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Flask API Endpoint
@app.route("/chat", methods=["POST"])
def chat_with_bot():
    data = request.get_json()
    query = data.get("message", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    retrieved_docs = retriever.get_relevant_documents(query)

    if not retrieved_docs:
        return jsonify({"reply": "No relevant information found in the database."})

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    response = llm.invoke(
        f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"
    )

    return jsonify({"reply": response.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
