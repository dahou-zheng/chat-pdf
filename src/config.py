import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found! Did you forget to set it in your .env file?")

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
TEST_PDFS_DIR = os.path.join(BASE_DIR, "test_data")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
VECTOR_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
META_PATH = os.path.join(VECTOR_STORE_DIR, "meta.json")

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:latest"


DEFAULT_TOP_K = 10
DEFAULT_BATCH_SIZE = 16




