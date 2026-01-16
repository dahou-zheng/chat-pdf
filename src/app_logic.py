import streamlit as st
from src.config import EMBEDDING_MODEL
from src.generator import load_client, test_connection
import src.vector_store
import src.embedding
import src.document_processor

state = st.session_state


def page_configuration():
    st.set_page_config(
        page_title="Chat with PDFs",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def document_management():
    st.header("Document Management")
    uploaded_file = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        help="Supports PDF format documents",
        accept_multiple_files=True
    )


def model_configuration():
    st.header("OpenAI Model Configuration")
    st.selectbox(
        "Embedding Model",
        options=[EMBEDDING_MODEL],
        disabled=True,
        help="Due to budget constraints, this is the only embedding model available."
    )
    model_option = st.selectbox(
        label="Select Language Model",
        options=["gpt-4o-mini", "gpt-4.1-nano", "gpt-5-nano"],
        help=("Choose the language model for generating responses.\n"
              "gpt-4o-mini is the standard choice; gpt-4.1-nano is faster; choose gpt-5-nano for reasoning")
    )


def connection_checker():
    if st.button("Test Connection", use_container_width=True):
        try:
            if "client" in state:
                test_connection(state.client)
                st.success("Connection successful!")
            else:
                st.error("Error loading client.")
        except Exception as e:
            st.error(f"Connection failed: {e}")


def index_status():
    st.subheader("Index Status")
    if "index_manager" in state:
        st.info(f"**Vector Count**: {state.index_manager.index.ntotal}\n\n"
                f"**Processed Files**: {len(state.index_manager.meta['files'])}")
        if state.index_manager.meta['files']:
            for file_name in state.index_manager.meta["files"]:
                st.text(f"• {file_name}")
    else:
        st.warning("No index created yet, please upload documents")


def initialize_state():
    if "client" not in state:
        try:
            state.client = load_client()
        except Exception as e:
            st.error(f"Error loading client: {e}")


def question_and_answer():
    if "index_manager" not in state or not state.index_manager.meta["files"]:
        st.info("Please upload and process documents on the left first, then you can ask questions")
        return

    st.subheader("Ask a Question")
    question = st.text_input(
        "Enter your question",
        placeholder="e.g. Explain statistical language models",
        key="question_input"
    )

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        submit_button = st.button("Submit", type="primary", use_container_width=True)

    with col2:
        use_reranking = st.checkbox("Enable Advanced Retrieval", value=True,
                                    help="Retrieve top 20, generate 3 query variations, rerank with RRF to get top results")

    with col3:
        if use_reranking:
            top_k = st.slider("Final documents (after reranking)", min_value=3, max_value=20, value=5, step=1)
        else:
            top_k = st.slider("Number of documents to retrieve", min_value=3, max_value=20, value=5, step=1)
