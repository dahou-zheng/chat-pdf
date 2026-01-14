from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple
import faiss
from embedding import embedding
from document_processor import _load_local_documents, split_documents_to_text_chunks
import numpy as np
from config import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIM,
    VECTOR_INDEX_PATH,
    META_PATH,
    DEFAULT_TOP_K,
    TEST_PDFS_DIR,
)


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

    def load_index(self) -> faiss.Index:
        return faiss.read_index(self.index_path)

    def save_index(self) -> None:
        # in case the config changes, and the new directory doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

    def load_meta(self) -> Dict[int, str]:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}

    def save_meta(self) -> None:
        # in case the config changes, and the new directory doesn't exist
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as file:
            # indent=2 for readability, ensure_ascii=False for unicode support
            json.dump({str(index): text for index, text in self.meta.items()}, file, ensure_ascii=False, indent=2)

    def save(self) -> None:
        self.save_index()
        self.save_meta()

    def add_texts(self, texts: List[str]) -> Tuple[faiss.Index, Dict[int, str]]:
        """
        Add new texts to the FAISS index.
        """
        if not texts:
            return

        vectors = _embed_texts(texts)
        faiss.normalize_L2(vectors)  # in-place L2 normalization

        start_id = (max(self.meta.keys()) + 1) if self.meta else 0
        ids = np.arange(start_id, start_id + len(texts)).astype("int64")

        self.index.add_with_ids(vectors, ids)
        for i, t in zip(ids.tolist(), texts):
            self.meta[i] = t

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, object]]:
        """
        Perform a top k similarity search in the FAISS index.
        """
        q_vec = _embed_texts([query])
        faiss.normalize_L2(q_vec)

        scores, ids = self.index.search(q_vec, top_k)  # a KNN search
        out: List[Dict[str, object]] = []
        for score, _id in zip(scores[0].tolist(), ids[0].tolist()):
            # when FAISS doesn't have enough results to fill top_k, it pads score and id with -1
            if _id != -1:
                out.append({"id": int(_id), "score": float(score), "text": self.meta.get(int(_id), "")})
        return out


if __name__ == "__main__":
    # 1) If index/meta files exist, load them; otherwise create new from PDF
    if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(META_PATH):
        print("Loading existing index...")
        index_manager = FaissManager(VECTOR_INDEX_PATH, META_PATH)
    else:
        print("Creating new index...")
        index_manager = FaissManager()
        documents = _load_local_documents(TEST_PDFS_DIR)
        text_chunks = split_documents_to_text_chunks(documents)
        index_manager.add_texts(text_chunks)
        index_manager.save()

    test_query = "What is natural language processing?"
    results = index_manager.search(test_query, top_k=5)
    print(f"Query: {test_query}")
    for r in results:
        print(f"--------------------------------\n{r['text']}\n")

