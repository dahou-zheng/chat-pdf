from __future__ import annotations

from io import BytesIO
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, IO
import streamlit as st


def load_documents(uploaded_files: List[IO]) -> List[Document]:
    """Load pdf documents from streamlit file_uploader and extract them to text documents."""
    documents = []
    for uploaded_file in uploaded_files:
        pdf_content = BytesIO(uploaded_file.read())
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        if text:
            documents.append(Document(page_content=text, metadata={"file_name": uploaded_file.name}))
    return documents


def split_documents_to_text_chunks(
    documents: List[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> List[str]:
    """
    Load list of documents and split into chunks using LangChain.

    Returns: List[str] (each item is a chunk of text)
    """

    textsplit = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = textsplit.split_documents(documents)
    return [chunk.page_content for chunk in chunks if chunk.page_content]


if __name__ == '__main__':
    # to test the above code, cmd: streamlit run src/document_processor.py and upload some pdfs (from test_data folder)
    with st.sidebar:
        st.header("Upload PDF(s)")
        files_from_user = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
    if files_from_user:
        documents = load_documents(files_from_user)
        print(f'number of pdf documents: {len(documents)}')
        text_chunks = split_documents_to_text_chunks(documents)
        print(f'number of text chunks: {len(documents)}')
        print('Each text chunk should be of type Document', type(text_chunks[0]))


