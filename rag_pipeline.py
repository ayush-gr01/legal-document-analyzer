
import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load the .env file from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from .pdf_processor import extract_text_from_pdf
from .chunking import split_documents
from .embeddings import load_embedding_model, compute_embeddings
from .vector_store import (
    EmbeddingWrapper,
    build_faiss_index,
    similarity_search,
)
from .prompts import build_prompt, SYSTEM_PROMPT

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is missing. Check your project's .env file."
    )

genai.configure(api_key=GEMINI_API_KEY)


# Model name configurable via env var, default to gemini-1.5-flash
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Initialize embedding model once
_embedding_model = load_embedding_model()
_embedder = EmbeddingWrapper(_embedding_model)

class RAGPipeline:
    """End‑to‑end RAG pipeline for a single session.

    The pipeline holds an in‑memory FAISS index and the associated documents.
    It can be reset when the user clears the chat or removes uploaded files.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, top_k: int = 4):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.documents: List[object] = []  # List of LangChain Document objects (chunks)
        self.vector_store = None
        self._initialized = False

    def _initialize_index(self, docs: List[object]):
        """Create FAISS index from chunked documents."""
        # Compute embeddings (LangChain vectorstore will compute again, but we keep for potential reuse)
        self.vector_store = build_faiss_index(docs, _embedder)
        self._initialized = True

    def add_pdf(self, pdf_path: str):
        """Process a PDF file and add its content to the RAG index.

        Raises
        ------
        ValueError
            If the PDF is scanned or contains no extractable text.
        """
        # Extract raw page documents
        page_docs = extract_text_from_pdf(pdf_path)
        # Chunk them
        chunked = split_documents(page_docs, self.chunk_size, self.chunk_overlap)
        # Initialize or update the FAISS index
        if not self._initialized:
            self.documents = chunked
            self._initialize_index(self.documents)
        else:
            # Append new docs and rebuild index (FAISS does not support incremental add easily without persistence)
            self.documents.extend(chunked)
            self._initialize_index(self.documents)
        return len(chunked)

    def _retrieve(self, query: str) -> List[object]:
        """Retrieve top‑k relevant chunks for the query."""
        if not self._initialized:
            return []
        return similarity_search(self.vector_store, query, top_k=self.top_k)

    def answer_query(self, question: str) -> dict:
        """Generate an answer using Gemini with retrieved context.

        Returns a dictionary with ``answer`` (string) and ``sources`` (list of dicts).
        """
        retrieved_docs = self._retrieve(question)
        if not retrieved_docs:
            return {
                "answer": "I could not find sufficient information in the uploaded documents to answer this question.",
                "sources": [],
            }

        # Prepare context passages with citations
        contexts = []
        sources = []
        for doc in retrieved_docs:
            citation = f"[{doc.metadata.get('source', 'unknown')} - page {doc.metadata.get('page', '?')}]"
            passage = f"{citation}\n{doc.page_content.strip()}"
            contexts.append(passage)
            sources.append({"source": doc.metadata.get('source'), "page": doc.metadata.get('page'), "content": doc.page_content.strip()})

        prompt = build_prompt(question, contexts)
        # Call Gemini API
        try:
            response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
            answer_text = response.text.strip()
        except Exception as e:
            answer_text = f"Error generating answer: {e}"

        return {"answer": answer_text, "sources": sources}

    def clear(self):
        """Reset the pipeline for a new session."""
        self.documents = []
        self.vector_store = None
        self._initialized = False
