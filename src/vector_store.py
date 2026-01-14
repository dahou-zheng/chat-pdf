from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple
import faiss
from embedding import embedding
from document_processor import split_documents_to_text_chunks
import numpy as np
from config import DEFAULT_BATCH_SIZE, EMBEDDING_DIM, VECTOR_INDEX_PATH, META_PATH


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """
    Yield successive n-sized chunks from items for batch processing.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _embed_texts(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """
    Returns float32 matrix: (n, EMBEDDING_DIM)
    """
    vectors: List[np.ndarray] = []
    for batch in _batched(texts, batch_size=batch_size):
        try:
            batch_vecs = embedding(batch)
        except Exception as e:
            raise RuntimeError(
                "Embedding request failed. Check `EMBEDDING_URL` in `embedding.py` "
                "and that the service is reachable."
            ) from e

        arr = np.asarray(batch_vecs, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Expected embeddings shape (n, {EMBEDDING_DIM}), got {arr.shape}")
        vectors.append(arr)

    vectors = np.vstack(vectors).astype("float32", copy=False)
    return vectors


class FaissManager:
    def __init__(self, index_path: str = VECTOR_INDEX_PATH, meta_path: str = META_PATH):
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            self.meta = self.load_meta()
        else:  # create new index
            self.index = FaissManager.create_index()
            self.meta: Dict[int, str] = {}

    @staticmethod
    def create_index(dim: int = EMBEDDING_DIM) -> faiss.IndexFlatIP:
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    def load_meta(self) -> Dict[int, str]:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}

    def save_meta(self):
        # in case the config changes, and the new directory doesn't exist
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as file:
            # indent=2 for readability, ensure_ascii=False for unicode support
            json.dump({str(index): text for index, text in self.meta.items()}, file, ensure_ascii=False, indent=2)






if __name__ == "__main__":
    # 1) If index/meta files exist, load them; otherwise create new from PDF
    if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(META_PATH):
        print("Loading existing index...")
        index_manager = FaissManager(VECTOR_INDEX_PATH, META_PATH)
    else:
        print("Creating new index...")
        index_manager = FaissManager()

