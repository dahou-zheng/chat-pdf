from __future__ import annotations

import os
from io import BytesIO
import fitz  # PyMuPDF
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, IO, Callable, Iterable
import streamlit as st
from src.config import TEST_PDFS_DIR


def load_documents(uploaded_files: Iterable[IO]) -> List[Document]:
    """Load pdf documents from streamlit file_uploader and extract them to text documents."""
    documents = []
    for uploaded_file in uploaded_files:
        pdf_content = BytesIO(uploaded_file.read())
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        if text:
            documents.append(Document(page_content=text, metadata={"file_name": uploaded_file.name}))
    return documents


def _load_local_documents(pdf_directory: str) -> List[Document]:
    """
    For testing: Load multiple PDF files directly from a local folder.

    Returns the same type as load_documents → List[Document]
    """
    documents = []
    pdf_files = os.listdir(pdf_directory)

    if not pdf_files:
        print(f"No PDF files found in '{pdf_directory}'")
        return documents

    for file_name in pdf_files:
        try:
            file_path = os.path.join(pdf_directory, file_name)
            print(file_path)
            with fitz.open(file_path) as doc:
                text = "".join(page.get_text() for page in doc)
            if text:
                documents.append(Document(page_content=text, metadata={"file_name": file_name}))
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print(f"Loaded {len(documents)} document(s) from '{pdf_directory}'")
    return documents


def num_tokens_from_string(string: str) -> int:
    """
    Use token length instead of string length as a better text splitting method.

    Returns the number of tokens in a text string.
    """
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


def split_documents_to_text_chunks(
    documents: List[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    length_function: Callable[[str], int] = num_tokens_from_string,
) -> List[str]:
    """
    Load list of documents and split into chunks using LangChain.

    Returns: List[str] (each item is a chunk of text)
    """

    textsplit = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function
    )
    chunks = textsplit.split_documents(documents)
    return chunks


if __name__ == '__main__':
    # to test the above code with streamlit, set test_with_streamlit = True, and
    # cmd: streamlit run src/document_processor.py and upload some pdfs (from test_data folder)
    test_with_streamlit = False
    if test_with_streamlit:
        with st.sidebar:
            st.header("Upload PDF(s)")
            files_from_user = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
        if files_from_user:
            documents = load_documents(files_from_user)
            print(f'number of pdf documents: {len(documents)}')
            text_chunks = split_documents_to_text_chunks(documents, length_function=num_tokens_from_string)
            print(f'number of text chunks: {len(text_chunks)}')
    else:
        documents = _load_local_documents(TEST_PDFS_DIR)
        print(f'number of pdf documents: {len(documents)}')
        chunks_by_token_length = split_documents_to_text_chunks(documents, length_function=num_tokens_from_string)
        print(f'number of text chunks chunked by token length: {len(chunks_by_token_length)}')
        chunks_by_str_length = split_documents_to_text_chunks(documents, length_function=len)
        print(f'number of text chunks chunked by string length: {len(chunks_by_str_length)}')



