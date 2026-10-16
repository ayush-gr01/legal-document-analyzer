
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
import numpy as np


class EmbeddingWrapper:
    """Adapt SentenceTransformer to LangChain's embedding interface."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        return embedding.tolist()
    
    def __call__(self, text: str) -> List[float]:
        """Allow FAISS to call the embedding wrapper directly."""
        return self.embed_query(text)

def build_faiss_index(
    documents: List[Document],
    embedder: EmbeddingWrapper
) -> FAISS:
    """Create a FAISS vector store using the embedding object."""

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedder
    )
    return vector_store


def similarity_search(
    vector_store: FAISS,
    query: str,
    top_k: int = 4
) -> List[Document]:
    """Retrieve the most relevant documents."""

    return vector_store.similarity_search(query, k=top_k)
