# AI Legal Document Analyzer using LLM + RAG

## Project Overview

A beginner‑friendly Streamlit web app that lets users upload legal PDF documents, ask natural‑language questions, get concise answers grounded in the uploaded material, and extract key clauses.  The backend uses **Retrieval‑Augmented Generation (RAG)**:

1. PDF → text with page numbers (PyMuPDF)
2. Text → chunks (LangChain splitter)
3. Chunks → embeddings (sentence‑transformers)
4. Embeddings → FAISS vector store
5. User query → embedding → similarity search → retrieved passages
6. Prompt + passages → Gemini LLM → answer with citations

All secrets (Gemini API key, model names) are read from environment variables – never hard‑coded.

---

## Architecture & RAG Workflow

```
PDF Upload → pdf_processor.extract_text() → chunking.chunk_text() → embeddings.compute_embeddings() → vector_store.store()

User asks → rag_pipeline.answer_query(question)
    ├─ Retrieve relevant chunks (FAISS)
    └─ Build prompt (prompts.PROMPT_TEMPLATE)
        → Gemini LLM → answer + source citations
```

## Prerequisites

- Windows 10/11
- Python 3.11 (`python --version`)
- Git (optional, for cloning)

## Installation (Windows)

```powershell
# 1. Clone or copy the project folder
cd "C:\Users\Laptop\Desktop"
# 2. Create a virtual environment
python -m venv venv
# 3. Activate it
venv\Scripts\Activate.ps1
# 4. Install dependencies
pip install -r "legal document analyzer\requirements.txt"
# 5. Set up environment variables
copy "legal document analyzer\.env.example" "legal document analyzer\.env"
#   Edit the .env file and add your Gemini API key.
```

## Running the Application

```powershell
cd "legal document analyzer"
streamlit run app.py
```

Open the URL shown in the console (typically `http://localhost:8501`).

## Running Tests

```powershell
cd "legal document analyzer"
pytest
```

All tests are written to run **without** contacting the Gemini API – the LLM calls are mocked.

---

## Project Structure

```
legal-document-analyzer/
├─ app.py                     # Streamlit UI
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ README.md
├─ src/
│  ├─ __init__.py
│  ├─ pdf_processor.py        # PDF → text + page metadata
│  ├─ chunking.py             # Text splitters
│  ├─ embeddings.py           # Local sentence‑transformer embeddings
│  ├─ vector_store.py         # FAISS wrapper
│  ├─ rag_pipeline.py         # End‑to‑end query handling
│  └─ prompts.py              # Prompt template & system message
└─ tests/
   ├─ test_pdf_processor.py
   ├─ test_chunking.py
   └─ test_rag_pipeline.py
```

## Key Design Decisions

- **Local embeddings** – no external API costs, using `all-MiniLM-L6-v2` (fast & small).
- **FAISS‑CPU** – works on any machine, no GPU required.
- **Gemini LLM** – accessed via `google-generativeai`. The model name is configurable (`GEMINI_MODEL`).
- **Citation format** – source filename and page number are attached to each `Document` metadata and inserted into the answer prompt.
- **Safety notice** – a disclaimer is shown in the UI stating that the output is informational and **not** legal advice.

---

## Limitations & Future Work

- No OCR – scanned PDFs are rejected with a clear message.
- Clause extraction is rule‑based (simple keyword search). A more sophisticated approach could use LLM extraction.
- Vector index is kept in memory per Streamlit session – persisting across restarts would require saving the FAISS index to disk.

---

## Resume‑Ready Description

*Built an end‑to‑end Retrieval‑Augmented Generation (RAG) web application with Streamlit, LangChain, FAISS, and Google Gemini. Implemented PDF parsing, chunking, local embeddings, similarity search, prompt engineering, and a clean UI for legal‑document question answering and clause extraction.*
