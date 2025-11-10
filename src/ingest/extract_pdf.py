import os
from pdfminer.high_level import extract_text
from docx import Document

def extract_pdf(path: str) -> str:
    """Extracts text from a PDF file using pdfminer.six."""
    try:
        text = extract_text(path)
        return text.strip()
    except Exception as e:
        print(f"[ERROR] PDF extraction failed for {path}: {e}")
        return ""

def extract_docx(path: str) -> str:
    """Extracts text from a DOCX file using python-docx."""
    try:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        print(f"[ERROR] DOCX extraction failed for {path}: {e}")
        return ""

def extract_txt(path: str) -> str:
    """Reads text from a plain TXT file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[ERROR] TXT extraction failed for {path}: {e}")
        return ""

def extract_any(path: str) -> str:
    """Dispatches extraction based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext == ".docx":
        return extract_docx(path)
    elif ext == ".txt":
        return extract_txt(path)
    else:
        print(f"[WARN] Unsupported file type: {ext}")
        return ""
