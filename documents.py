class Document:
    """Simple stand‑in for LangChain's Document class.

    Attributes
    ----------
    page_content: str
        The textual content of the document or chunk.
    metadata: dict
        Arbitrary metadata, typically containing ``source`` and ``page`` keys.
    """

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(page_content={self.page_content!r}, metadata={self.metadata!r})"
