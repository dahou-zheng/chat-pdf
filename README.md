# chat-pdf
An AI-powered document assistant that enables users to upload PDF files and ask questions. It uses Retrieval-Augmented Generation (RAG) to provide precise answers backed by exact quotes and paragraphs from the source document.

## Features

- **PDF Upload & Processing**: Upload and manage single or multiple PDF documents, with automatic parsing and indexing.
- **Document-Grounded Q&A**: Ask questions in natural language and receive answers based on the content of the uploaded documents.
- **Source Citations**: Answers include quoted text and paragraph-level references to the original documents.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with language model generation to provide context-aware responses.
- **Advanced Retrieval Techniques**: Applies query reformulation, query condensation, and reranking to improve retrieval relevance.

## Quick Start

### Prerequisites

- Python **3.13** (developed and tested with 3.13.5)
- All dependencies are listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dahou-zheng/chat-pdf.git
cd chat-pdf
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
# Add other required API keys as needed
```

4. Run the application:
```bash
streamlit run app.py
```

## Technology Stack

### Frontend
- **UI Framework**: Streamlit (interactive web interface, file upload, chat-like Q&A experience)

### Backend / Core Logic
- **Orchestration**: LangChain
- **LLM**: OpenAI GPT models
- **Embeddings**: OpenAI Embeddings
- **Vector Store**: FAISS (local in-memory / persistent index)
- **PDF Processing**: PyMuPDF (fitz) for extraction and text parsing
- **Language & Runtime**: Python 3.13 (or your exact version, e.g. 3.13.5)

## Project Structure

```
chat-pdf/
├── app.py                    # Streamlit entry point (frontend UI)
├── requirements.txt          # Python dependency list
├── .gitignore                # Git ignore configuration
├── data/                     # Persistent data storage
│   ├── test_data/            # Sample PDFs for development and testing
│   └── vector_store/         # FAISS index files and custom metadata
├── src/                      # Application source code
│   ├── __init__.py
│   ├── app_logic.py          # Core application workflow and orchestration
│   ├── config.py             # Global configuration and constants
│   ├── document_processor.py # PDF loading, parsing, and chunking
│   ├── embedding.py          # Text embedding generation
│   ├── generator.py          # LLM-based generation tasks (e.g. Q&A, reformulation, condesation, reranking)
│   └── vector_store.py       # Vector index creation, loading, and metadata management
└── README.md                 # Project documentation
```
