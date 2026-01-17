from __future__ import annotations
import streamlit as st
from typing import List, IO, Any
from src.config import EMBEDDING_MODEL
from src.generator import (
    load_client,
    test_connection,
    generate_query_reformulations,
    reciprocal_rank_fusion,
    format_context,
    generate_answer,
    condense_multi_turn_query
)
from src.vector_store import FaissManager
from src.document_processor import load_documents, split_documents_to_text_chunks

state = st.session_state


def page_configuration():
    st.set_page_config(
        page_title="Chat with PDFs",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def update_index(files: List[IO]):
    files = {file.name: file for file in files}
    for file_name in list(state.index_manager.meta["files"].keys()):
        if file_name in files:
            del files[file_name]
        else:
            state.index_manager.delete_file(file_name)
    documents = load_documents(list(files.values()))
    chunks = split_documents_to_text_chunks(documents)
    state.index_manager.add_chunks(chunks)


def init_index(files: List[IO]):
    state.index_manager = FaissManager()
    state.index_manager.reset()

    documents = load_documents(files)
    chunks = split_documents_to_text_chunks(documents)
    state.index_manager.add_chunks(chunks)


def document_management():
    st.header("Document Management")

    if "index_manager" in state:
        if st.button("Clear Index", use_container_width=True, help="Clear the current index and start fresh"):
            state.index_manager.reset()
            st.success("Index cleared successfully!")
            st.rerun()

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        help="Supports PDF format documents",
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Sync Document Library", type="primary", use_container_width=True):
            try:
                with st.spinner("Processing document..."):
                    if "index_manager" in state:
                        update_index(uploaded_files)
                    else:
                        init_index(uploaded_files)
                st.success(f"Document Library updated")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing documents: {e}")


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
    state.model = model_option


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


def search_and_answer(query: str, top_k: int, use_reranking: bool, rerank_top_k: int) -> Any:
    if use_reranking:
        all_search_results = []

        # 1. Initial retrieval with original query
        with st.spinner("Initial retrieval with original query..."):
            results = state.index_manager.search(query, top_k=top_k)
            if results:
                all_search_results.append(results)

        # 2. Generate query reformulations
        try:
            with st.spinner("Generating answer..."):
                reformulations = generate_query_reformulations(query, state.client, state.model, num_reformulations=3)
        except Exception as e:
            st.error(f"Error generating query reformulations: {e}. Using original query only.")
            reformulations = []

        # 3. Retrieve with each reformulated query
        if reformulations:
            for i, reformed_query in enumerate(reformulations, 1):
                with st.spinner(f"Retrieving with reformulation {i}/3..."):
                    results = state.index_manager.search(reformed_query, top_k=top_k)
                    if results:
                        all_search_results.append(results)

        # 4. Apply Reciprocal Rank Fusion (RRF) reranking
        if all_search_results:
            with st.spinner("Reranking results using Reciprocal Rank Fusion..."):
                search_results = reciprocal_rank_fusion(all_search_results, top_k=rerank_top_k)

                # Store reformulations in session state for display
                if reformulations:
                    state.last_reformulations = reformulations
    else:
        # Simple retrieval without reranking
        with st.spinner("Searching relevant documents..."):
            search_results = state.index_manager.search(query, top_k=top_k)

    if not search_results:
        return "No relevant documents found"
    state.last_search_results = search_results

    # 5. Build context from top results
    context = format_context(search_results)

    # 6. Generate answer
    try:
        with st.spinner("Generating answer..."):
            answer = generate_answer(context, query, state.client, state.model)
            return answer
    except Exception as e:
        raise RuntimeError(f"Error generating answer: {e}")


def report_reformulations():
    st.subheader("Query Reformulations")
    for i, reform in enumerate(state.last_reformulations, 1):
        st.text(f"{i}. {reform}")


def report_search_results():
    st.subheader("Relevant Document Chunks")
    for i, result in enumerate(state.last_search_results, 1):
        score_type = "RRF Score" if 'rrf_score' in result else "Similarity"
        score_value = result.get('rrf_score', result.get('score', 0))
        file_name = result.get('file_name', '')
        with st.expander(f"Chunk {i} ({score_type}: {score_value:.4f}), from file: {file_name}"):
            st.text(result['text'])


def clear_chat_history():
    with st.spinner("Starting new chat..."):
        state.chat_history = []
        state.pop("last_reformulations", None)
        state.pop("last_search_results", None)
    st.success("History cleared.")


def display_chat_history():
    for message in state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def question_and_answer():
    if "index_manager" not in state or not state.index_manager.meta["files"]:
        st.info("Please upload and process documents on the left first, then you can ask questions")
        return
    if "chat_history" not in state:
        state.chat_history = []

    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
    with col1:
        if st.button("Start a new chat", type="primary", use_container_width=True):
            clear_chat_history()
    with col2:
        use_reranking = st.checkbox("Enable Advanced Retrieval", value=True,
                                    help="Original query + 3 query variations, each retrieve k results, "
                                         "rerank with RRF to get top results")
    with col3:
        if use_reranking:
            top_k = st.slider("Number of documents per retrieval", min_value=3, max_value=20, value=5, step=1)
        else:
            top_k = st.slider("Number of documents", min_value=3, max_value=20, value=5, step=1)
    with col4:
        if use_reranking:
            rerank_top_k = st.slider("Number of documents after reranking", min_value=3, max_value=20, value=5, step=1)
        else:
            rerank_top_k = 0

    question = st.chat_input(placeholder="Enter your question")
    if question:
        try:
            if state.chat_history:
                condensed_question = condense_multi_turn_query(state.chat_history, question, state.client, state.model)
                answer = search_and_answer(condensed_question, top_k, use_reranking, rerank_top_k)
            else:
                answer = search_and_answer(question, top_k, use_reranking, rerank_top_k)
        except Exception as e:
            st.error(f"Error generating answer: {e}")

        if answer:
            state.chat_history.append({"role": "user", "content": question})
            state.chat_history.append({"role": "assistant", "content": answer})

            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            if "last_reformulations" in state:
                with col1:
                    report_reformulations()
            if "last_search_results" in state:
                with col2:
                    report_search_results()

        display_chat_history()


