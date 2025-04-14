# AI-Powered Hybrid Search Chatbot


## 📚 Project Overview
This project implements an AI-powered chatbot that utilizes hybrid search techniques to provide answers based on a given text dataset. The chatbot leverages advanced retrieval methods to find relevant information and generate responses using a language model.

## 🛠️ Use Case
The AI chatbot can be used in various scenarios, including:
- **Customer Support**: Answering frequently asked questions based on a knowledge base.
- **Educational Tools**: Assisting students by providing answers to questions related to study materials.
- **Research Assistance**: Helping researchers quickly find relevant literature or data.

## 🔍 Techniques of Retrieval
The chatbot employs a combination of the following retrieval techniques:
1. **Dense Retrieval**: Utilizes embeddings from a language model to encode and retrieve relevant documents based on semantic similarity.
2. **Sparse Retrieval**: Implements a BM25 encoder to perform traditional keyword-based search, enhancing the accuracy of retrieved results.
3. **Hybrid Search**: Combines both dense and sparse retrieval methods to leverage the strengths of each approach, providing more comprehensive and relevant search results.

## 🛠️ Tools and Libraries
This project uses several tools and libraries to implement the chatbot:
- **Streamlit**: For creating the user interface of the chatbot.
- **LangChain**: A framework for building applications with language models, facilitating document loading, text splitting, and retrieval operations.
- **Pinecone**: A vector database for managing and searching embeddings efficiently.
- **Hugging Face Transformers**: For generating embeddings using pre-trained models.
- **BM25**: A traditional information retrieval algorithm to enhance search capabilities.
- **Groq**: A powerful language model for generating responses based on the retrieved context.
