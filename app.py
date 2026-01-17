import streamlit as st
from src.app_logic import (
    page_configuration,
    initialize_state,
    question_and_answer,
    document_management,
    model_configuration,
    connection_checker,
    index_status,
)


page_configuration()

st.title("Chat with PDFs")
st.markdown("---")
initialize_state()
question_and_answer()

with st.sidebar:
    document_management()
    st.markdown("---")
    model_configuration()
    st.markdown("---")
    index_status()
    st.markdown("---")
    connection_checker()


