import os
import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------
# Load PDFs using PyMuPDF
# -------------------------------
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        pdf_content = BytesIO(uploaded_file.read())
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        if text:
            documents.append(Document(page_content=text, metadata={"file_name": uploaded_file.name}))
    return documents


# -------------------------------
# Vector Store Setup
# -------------------------------
def get_vector_store(file_content):
    # Load and split documents
    documents = load_documents(file_content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents=documents)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model="mistral:latest")
    return FAISS.from_documents(documents=chunks, embedding=embeddings)


# -------------------------------
# Format retrieved documents
# -------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -------------------------------
# Build RAG Chain
# -------------------------------
def get_rag_chain(vector_store):
    # Set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Create prompt template
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template=template)

    # Initialize LLM
    llm = OllamaLLM(model="mistral:latest")

    # Build RAG chain
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


# -------------------------------
# Generate Response
# -------------------------------
def get_response(user_input):
    # Get RAG chain from session state vector store
    rag_chain = get_rag_chain(st.session_state.vector_store)

    # Invoke chain with user input
    response = rag_chain.invoke(user_input)
    return response


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.header("📚 Chat with your PDFs using mistral:latest (Ollama)")

with st.sidebar:
    st.header("📄 Upload PDF(s)")
    uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # this if statement should be ignored because it only allows users to upload documents once and never update anymore
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store(uploaded_files)

    user_input = st.chat_input("Type your message...")
    if user_input and user_input.strip() != "":
        response = get_response(user_input)
        with st.chat_message("User"):
            st.write(user_input)
        with st.chat_message("AI"):
            st.write(response)
