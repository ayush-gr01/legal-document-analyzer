from sentence_transformers import SentenceTransformer
import os
from typing import List
import numpy as np
from langchain_core.documents import Document


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model specified by ``EMBEDDING_MODEL`` env var.

    Defaults to ``all-MiniLM-L6-v2`` if not set.
    """
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def compute_embeddings(model: SentenceTransformer, documents: List[Document]) -> np.ndarray:
    """Compute embeddings for a list of ``Document`` objects.

    Returns a 2‑D NumPy array where each row corresponds to a document.
    """
    texts = [doc.page_content for doc in documents]
    # ``model.encode`` returns a list of ndarrays; set ``convert_to_numpy`` for efficiency.
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings
