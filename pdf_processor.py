import fitz
from langchain_core.documents import Document
import os

def extract_text_from_pdf(file_path: str) -> list[Document]:
    """Extract plain‑text from a PDF and preserve page numbers.

    Parameters
    ----------
    file_path: str
        Absolute path to the PDF file.

    Returns
    -------
    List[Document]
        A list of ``Document`` objects, one per page, with ``metadata``
        containing ``{"source": filename, "page": page_number}``.

    Raises
    ------
    ValueError
        If the PDF does not contain extractable text (e.g., scanned image PDF).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pdf = fitz.open(file_path)
    documents = []
    filename = os.path.basename(file_path)
    for page_number in range(pdf.page_count):
        page = pdf.load_page(page_number)
        text = page.get_text("text").strip()
        if not text:
            # If any page is empty, consider the whole PDF likely scanned.
            raise ValueError(
                "No extractable text found on page {}. "
                "This appears to be a scanned PDF; OCR is required.".format(page_number + 1)
            )
        doc = Document(page_content=text, metadata={"source": filename, "page": page_number + 1})
        documents.append(doc)
    pdf.close()
    return documents
