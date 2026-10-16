import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pdf_processor import extract_text_from_pdf

def test_extract_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("nonexistent.pdf")
