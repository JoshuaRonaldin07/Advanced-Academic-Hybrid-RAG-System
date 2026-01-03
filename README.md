# Advanced-Academic-Hybrid-RAG-System
A high-performance, locally-hosted Retrieval-Augmented Generation (RAG) system designed for academic research. This tool allows users to have a conversation with complex PDF documents (research papers, textbooks, notes) with industry-grade precision.
Key Features
Hybrid Retrieval: Combines Semantic Search (ChromaDB + HuggingFace Embeddings) with Lexical Search (BM25) to ensure both conceptual and keyword accuracy.

Two-Stage Pipeline: Implements a Re-ranker (BGE-Reranker) to refine the top results before they reach the LLM, significantly reducing hallucinations.

Local & Private: Powered by Ollama (Llama-3/Phi-3). Your documents never leave your local machine, ensuring 100% data privacy.

Academic Citation: Displays exact source chunks used to generate answers for fact-checking.

Technical Stack
Orchestration: LangChain (LCEL)

LLM: Ollama (Llama-3 / Phi-3)

Vector Store: ChromaDB

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Re-ranker: BAAI Cross-Encoder (bge-reranker-base)

Frontend: Streamlit

Architecture
The system follows a sophisticated "Retriever-Reranker" architecture:

Document Ingestion: PDFs are split using RecursiveCharacterTextSplitter.

Ensemble Retrieval: The query is sent to both a Vector Store (for meaning) and a BM25 index (for keywords).

RRF Merging: Results are combined using Reciprocal Rank Fusion.

Contextual Compression: A Cross-Encoder re-scores the top 10 results to find the 3 most relevant chunks.

Generation: The refined context is sent to the local LLM for a final grounded answer.

Installation & Setup
1. Prerequisites
Install Ollama

Pull your preferred model:

Bash

ollama pull phi3:mini

2. Clone and Install Dependencies
Bash

git clone https://github.com/your-username/academic-hybrid-rag.git
cd academic-hybrid-rag
pip install -r requirements.txt

3. Run the Application
Bash

streamlit run RAG.py
