import pytest
from src.chunking import split_documents
from langchain_core.documents import Document

def test_split_documents_preserves_metadata():
    docs = [Document(page_content="This is a test text that will be split into multiple chunks.", metadata={"source": "test.pdf", "page": 1})]
    chunks = split_documents(docs, chunk_size=20, chunk_overlap=5)
    # Expect multiple chunks
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1
        assert "chunk_index" in chunk.metadata
