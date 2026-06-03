"""
services/layer1/universal_parser.py — Universal File Parser

Fixes Error 5 (Batch 4): This file was completely empty.
Routes the uploaded file to the correct parser based on the detected InputType.
- CSV/XLSX → pandas DataFrame → raw tabular data dict
- PDF       → pdfplumber text extraction
- TXT/MD/DOCX → plain text extraction
"""
import os
from typing import Union

from services.layer1.file_detector import InputType


def parse_file(filepath: str, input_type: InputType) -> dict:
    """
    Parses the uploaded file into a normalised dict:

    For CSV/tabular files:
        {
            "type": "tabular",
            "columns": [...],
            "rows": [[...], ...],
            "row_count": int
        }

    For text/document files:
        {
            "type": "text",
            "content": "<full extracted text>",
            "char_count": int
        }

    Parameters
    ----------
    filepath : str
        Absolute path to the uploaded file on disk.
    input_type : InputType
        The detected type from file_detector.detect_input_type().

    Returns
    -------
    dict
        Normalised content dictionary consumed by the downstream classifier
        and extractor services.
    """
    if input_type == InputType.CSV:
        return _parse_tabular(filepath)
    elif input_type in (InputType.PDF, InputType.TEXT):
        return _parse_text(filepath, input_type)
    else:
        raise ValueError(f"Cannot parse unsupported InputType: {input_type}")


# ---------------------------------------------------------------------------
# Tabular Parser (CSV, TSV, XLSX)
# ---------------------------------------------------------------------------

def _parse_tabular(filepath: str) -> dict:
    """Uses pandas to read CSV/XLSX files into a serialisable dict."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError(
            "pandas is required for CSV parsing. "
            "Install it with: pip install pandas openpyxl"
        )

    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        sep = "\t" if ext == ".tsv" else ","
        df = None
        for enc in ("utf-8", "utf-16", "utf-8-sig", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if df is None:
            df = pd.read_csv(filepath, sep=sep, encoding="utf-8", errors="replace")

    return {
        "type": "tabular",
        "columns": list(df.columns),
        "rows": df.values.tolist(),
        "row_count": len(df),
    }


# ---------------------------------------------------------------------------
# Text / Document Parser (PDF, TXT, DOCX, MD)
# ---------------------------------------------------------------------------

def _parse_text(filepath: str, input_type: InputType) -> dict:
    """Extracts raw text from PDF, DOCX, or plain text files."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        content = _extract_pdf(filepath)
    elif ext in (".docx", ".doc"):
        content = _extract_docx(filepath)
    else:
        # Plain text / markdown — try utf-8, utf-16, latin-1
        content = None
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                with open(filepath, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if content is None:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

    return {
        "type": "text",
        "content": content,
        "char_count": len(content),
    }


def _extract_pdf(filepath: str) -> str:
    """Extracts text from a PDF using pdfplumber (best for structured PDFs)."""
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError(
            "pdfplumber is required for PDF parsing. "
            "Install it with: pip install pdfplumber"
        )

    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _extract_docx(filepath: str) -> str:
    """Extracts text from a DOCX file using python-docx."""
    try:
        import docx
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX parsing. "
            "Install it with: pip install python-docx"
        )

    doc = docx.Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
