import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Streamlit Page Config ---
st.set_page_config(page_title="Local Academic RAG (Llama-3)", layout="wide")
st.title("🤖 Local Hybrid RAG Assistant")
st.markdown("This system uses **Ollama (Llama-3)** and **Hybrid Search** to analyze your PDFs locally.")

# --- Initialize Session States ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload Academic PDF", type="pdf")
    
    if st.button("Initialize Local Index") and uploaded_file:
        with st.spinner("Processing PDF locally... This may take a minute."):
            # Save file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 1. Document Processing
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            splits = splitter.split_documents(docs)
            
            # 2. Setup Retrievers (Semantic + Keyword)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(splits, embeddings)
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            
            bm25_retriever = BM25Retriever.from_documents(splits)
            bm25_retriever.k = 10
            
            # 3. Hybrid Ensemble + Re-ranking
            ensemble = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], 
                weights=[0.4, 0.6]
            )
            
            # High-impact step: Cross-Encoder Re-ranking
            reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
            
            st.session_state.retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=ensemble
            )
            st.success("Local Database Ready!")

# --- Main Chat Interface ---
if st.session_state.retriever:
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your paper..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Local LLM Logic via Ollama
        llm = ChatOllama(model="llama3", temperature=0)
        
        sys_prompt = (
            "You are an academic researcher. Answer the question strictly using the provided context. "
            "If the answer is not in the context, say you don't know.\n\n"
            "Context: {context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "{input}")])
        
        # Build the chain
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(st.session_state.retriever, qa_chain)
        
        with st.chat_message("assistant"):
            with st.spinner("Llama-3 is thinking..."):
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                
                # Show citations
                with st.expander("Show Sources"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Source {i+1}:** {doc.page_content}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("← Please upload a PDF and click 'Initialize' to start chatting.")