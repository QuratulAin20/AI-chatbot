import os
import streamlit as st
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

# Streamlit Page Configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("📚 AI-Powered Hybrid Search Chatbot")
st.write("Ask any question, and I'll find the best answer for you!")

# One-time API key input
if "PINECONE_API_KEY" not in st.session_state or "GROQ_API_KEY" not in st.session_state:
    st.subheader("🔑 Enter Your API Keys (One-Time Setup)")
    pinecone_key = st.text_input("Pinecone API Key", type="password")
    groq_key = st.text_input("Groq API Key", type="password")

    if st.button("Save API Keys"):
        if pinecone_key and groq_key:
            st.session_state["PINECONE_API_KEY"] = pinecone_key
            st.session_state["GROQ_API_KEY"] = groq_key
            st.success("API Keys saved! You can now ask questions.")
            st.rerun()
        else:
            st.error("Both API keys are required!")

# Ensure API keys exist before proceeding
if "PINECONE_API_KEY" in st.session_state and "GROQ_API_KEY" in st.session_state:
    PINECONE_API_KEY = st.session_state["PINECONE_API_KEY"]
    GROQ_API_KEY = st.session_state["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "hybrid-search-langchain-pinecone"

    # Create or connect to Pinecone index
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

    # Chatbot function (Cleaned response)
    def chat_with_bot(query):
        retrieved_docs = retriever.get_relevant_documents(query)
        
        if not retrieved_docs:
            return "No relevant information found in the database."

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        response = llm.invoke(
            f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"
        )

        return response.content  # Extract only the generated answer

    # Chatbot UI
    user_input = st.text_input("💬 Ask me anything:")
    
    if user_input:
        st.write("🔍 Searching the database...")
        response = chat_with_bot(user_input)
        st.write("💡 **Chatbot Response:**")
        st.success(response)
