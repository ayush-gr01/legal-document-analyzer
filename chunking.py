
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller overlapping text chunks."""

    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)

    # Fallback for when LangChain splitters are unavailable
    chunks = []

    for document in documents:
        text = document.page_content

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Preserve the original document's metadata
            from langchain_core.documents import Document

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=document.metadata
                )
            )

            start += chunk_size - chunk_overlap

    return chunks
