# RAG-langchain-Gemini
RAG using Gemini
Banking RAG API

A Retrieval-Augmented Generation (RAG) application that answers banking-related questions using a banking knowledge base. The application uses LangChain to orchestrate the RAG pipeline, ChromaDB for vector storage and retrieval, Hugging Face for embeddings, Google Gemini for answer generation, and FastAPI to expose REST endpoints.

Tech Stack

Language: Python

API Framework: FastAPI

LLM Orchestration: LangChain

Embedding Model: Hugging Face all-MiniLM-L6-v2

Vector Database: ChromaDB

LLM: Google Gemini API

Server: Uvicorn

API Testing: Postman

Features

Banking document ingestion and text splitting

Text embedding generation

Semantic search using ChromaDB

Context-based answer generation using Google Gemini

REST API for banking questions

Source document information in API responses

Interactive API documentation using Swagger UI

Architecture

flowchart TD
    A[Banking Document: banking.txt] --> B[Load Document]
    B --> C[Split Text into Chunks]
    C --> D[Generate Embeddings]
    D --> [(ChromaDB Vector Store)]E

    F[User Question] --> G[FastAPI POST /ask]
    G --> H[LangChain RAG Pipeline]
    H --> I[Generate Question Embedding]
    I --> J[Similarity Search]
    E --> J
    J --> K[Retrieve Relevant Chunks]
    K --> L[Build Prompt with Context]
    F --> L
    L --> M[Google Gemini]
    M --> N[Generate Answer]
    N --> O[Return Answer and Sources]
    O --> P[API Response]

Complete Working Pipeline

1. Document Ingestion

The application reads banking information from data/documents/banking.txt.

2. Text Splitting

The document is divided into smaller chunks to improve retrieval.

Chunk size: 500 characters

Chunk overlap: 50 characters

3. Embedding Generation

The Hugging Face all-MiniLM-L6-v2 model converts each text chunk into a numerical vector representing its semantic meaning.

4. Vector Storage

The generated embeddings and associated document content are stored in ChromaDB. The database persists locally so the application can reuse indexed documents after restarting.

5. User Question

The user submits a banking question through Postman or another API client.

6. Semantic Retrieval

FastAPI passes the question to the LangChain RAG pipeline. The retriever searches ChromaDB for the two most relevant document chunks.

7. Context Construction

LangChain combines the user's question with the retrieved text to create a prompt for the language model.

8. Answer Generation

Google Gemini receives the prompt and generates a natural-language answer based on the retrieved context.

The prompt instructs the model not to invent banking information when the answer is not available in the supplied documents.

9. API Response

FastAPI returns the generated answer and the retrieved source information as a JSON response.


