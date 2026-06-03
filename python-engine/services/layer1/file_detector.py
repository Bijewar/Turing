"""
services/layer1/file_detector.py

Provides file detection utilities.
Combines simple extension/sniffing routing (Mayank) 
and advanced in-memory Universal File Detection (Harsh).
"""

import os
import io
import re
import logging
from enum import Enum
from pathlib import Path

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. ORCHESTRATION UTILITIES (Mayank)
# ==============================================================================

class InputType(str, Enum):
    CSV = "csv"
    TEXT = "text"
    PDF = "pdf"
    UNKNOWN = "unknown"

# CSV-like extensions → Data Path (PC Algorithm)
CSV_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
# Text/document extensions → Text Path (Ontology Builder + LLM)
TEXT_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".doc"}

def detect_input_type(filepath: str) -> InputType:
    """
    Layer 1 Gate: Detects the input file type and routes to the correct path.
    - CSV/spreadsheet → Data Path (PC Algorithm for causal discovery)
    - PDF/text/doc   → Text Path (LLM for ontology building)
    """
    ext = Path(filepath).suffix.lower()

    if ext in CSV_EXTENSIONS:
        return InputType.CSV
    elif ext == ".pdf":
        return InputType.PDF
    elif ext in TEXT_EXTENSIONS:
        return InputType.TEXT
    else:
        # Attempt content sniffing for files without clear extensions
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(512)
            # Heuristic: if the first line has commas and looks like headers, treat as CSV
            first_line = sample.split("\n")[0]
            if first_line.count(",") >= 2 or first_line.count("\t") >= 2:
                return InputType.CSV
            return InputType.TEXT
        except OSError:
            return InputType.UNKNOWN

def validate_file_exists(filepath: str) -> bool:
    """Checks that the file exists and is non-empty before processing."""
    path = Path(filepath)
    return path.exists() and path.is_file() and path.stat().st_size > 0

# ==============================================================================
# 2. ADVANCED UNIVERSAL FILE DETECTOR (Harsh)
# ==============================================================================

class UniversalFileDetector:
    """
    Step 1: Universal File Detector.
    Determines if a file should route to the DATA PATH or TEXT PATH.
    """

    @staticmethod
    def analyze_file(file_bytes: bytes, filename: str) -> dict:
        """
        Analyzes the file in-memory and returns the classification.
        """
        max_mb = float(os.getenv("MAX_UPLOAD_MB", "50"))
        if len(file_bytes) > max_mb * 1024 * 1024:
            raise ValueError(f"File exceeds maximum allowed size of {max_mb} MB.")

        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        result = {
            "path": "UNKNOWN",
            "confidence": 0.0,
            "sample_preview": "",
            "details": ""
        }

        try:
            # Magic byte checks
            if file_bytes.startswith(b"%PDF"):
                return UniversalFileDetector._analyze_pdf(file_bytes)
            if file_bytes.startswith(b"PK\x03\x04") and (ext in ['docx', 'doc'] or ext not in ['xlsx', 'xls']):
                return {"path": "TEXT", "confidence": 0.9, "sample_preview": "DOCX/Archive Document detected", "details": "ZIP/DOCX magic bytes detected."}

            if ext in ['csv']:
                result = UniversalFileDetector._analyze_tabular(file_bytes, 'csv')
            elif ext in ['xlsx', 'xls']:
                result = UniversalFileDetector._analyze_tabular(file_bytes, 'excel')
            elif ext in ['json']:
                result = UniversalFileDetector._analyze_tabular(file_bytes, 'json')
            elif ext in ['pdf']:
                result = UniversalFileDetector._analyze_pdf(file_bytes)
            elif ext in ['txt', 'md']:
                # Sniff CSV content inside .txt if comma-separated
                first_lines = file_bytes[:1024].decode("utf-8", errors="replace").splitlines()
                if first_lines and first_lines[0].count(',') >= 2:
                    try:
                        return UniversalFileDetector._analyze_tabular(file_bytes, 'csv')
                    except Exception:
                        pass
                result = UniversalFileDetector._analyze_text(file_bytes)
            else:
                # Check for JSON content or CSV content
                try:
                    import json
                    text_candidate = file_bytes.decode("utf-8", errors="replace")
                    if text_candidate.strip().startswith('{') and text_candidate.strip().endswith('}'):
                        json.loads(text_candidate)
                        return UniversalFileDetector._analyze_tabular(file_bytes, 'json')
                except Exception:
                    pass

                # Attempt text analysis as best-effort
                result = UniversalFileDetector._analyze_text(file_bytes)
        except Exception as e:
            logger.error(f"Failed to analyse file '{filename}': {e}", exc_info=True)
            raise ValueError(
                f"File '{filename}' could not be analysed (ext='.{ext}'): {e}"
            ) from e

        return result

    @staticmethod
    def _analyze_tabular(file_bytes: bytes, format_type: str) -> dict:
        """Analyzes structured files using pandas to check numeric density."""
        df = None
        if format_type == 'csv':
            # Read only first 100 rows for speed with encoding fallbacks
            for enc in ('utf-8', 'utf-16', 'utf-8-sig', 'latin-1'):
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), nrows=100, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if df is None:
                df = pd.read_csv(io.BytesIO(file_bytes), nrows=100)
        elif format_type == 'excel':
            df = pd.read_excel(io.BytesIO(file_bytes), nrows=100)
        elif format_type == 'json':
            df = pd.read_json(io.BytesIO(file_bytes))
            df = df.head(100)
        
        if df is None or df.empty:
            return {"path": "TEXT", "confidence": 0.9, "sample_preview": "Empty data", "details": "Structured file was empty."}

        # Calculate ratio of numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        total_cols = len(df.columns)
        
        if total_cols == 0:
             return {"path": "TEXT", "confidence": 0.9, "sample_preview": "No columns", "details": "No columns found."}
             
        numeric_ratio = len(numeric_cols) / total_cols
        
        # If > 50% of columns are numbers, it's DATA
        is_data = numeric_ratio >= 0.5
        
        sample_text = df.head(3).to_string()
        if len(sample_text) > 200:
            sample_text = sample_text[:197] + "..."

        return {
            "path": "DATA" if is_data else "TEXT",
            "confidence": round(numeric_ratio if is_data else (1 - numeric_ratio), 2),
            "sample_preview": sample_text,
            "details": f"Tabular data: {len(numeric_cols)}/{total_cols} numeric columns."
        }

    @staticmethod
    def _analyze_text(file_bytes: bytes) -> dict:
        """Analyzes raw text to check if it's actually numbers separated by spaces/tabs."""
        text = None
        for enc in ('utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-8-sig', 'latin-1'):
            try:
                text = file_bytes.decode(enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if text is None:
            return {"path": "DATA", "confidence": 0.5, "sample_preview": "Binary data", "details": "Could not decode text, assuming binary data."}
            
        lines = text.splitlines()[:100]
        if not lines:
             return {"path": "TEXT", "confidence": 0.9, "sample_preview": "Empty text", "details": "Text file was empty."}

        # Count tokens
        numeric_tokens = 0
        word_tokens = 0
        
        preview_lines = []
        for line in lines:
            if len(preview_lines) < 3:
                preview_lines.append(line)
            tokens = line.split()
            for t in tokens:
                # Strip common punctuation
                clean_t = t.strip('.,;:"\'()[]{}')
                is_num = False
                try:
                    float(clean_t)
                    is_num = True
                except (ValueError, TypeError):
                    is_num = False

                if is_num:
                    numeric_tokens += 1
                elif clean_t.isalpha() or len(clean_t) > 0:
                    word_tokens += 1

        total = numeric_tokens + word_tokens
        if total == 0:
             return {"path": "TEXT", "confidence": 0.9, "sample_preview": "\n".join(preview_lines), "details": "No identifiable tokens."}
             
        numeric_ratio = numeric_tokens / total
        is_data = numeric_ratio >= 0.5 # Consistent with tabular path: requires majority of tokens to be numbers
        
        sample_preview = "\n".join(preview_lines)
        if len(sample_preview) > 200:
            sample_preview = sample_preview[:197] + "..."

        return {
            "path": "DATA" if is_data else "TEXT",
            "confidence": round(numeric_ratio if is_data else (1 - numeric_ratio), 2),
            "sample_preview": sample_preview,
            "details": f"Raw text: {numeric_tokens} numbers, {word_tokens} words."
        }

    @staticmethod
    def _analyze_pdf(file_bytes: bytes) -> dict:
        """Uses pdfplumber to check for data tables vs plain text paragraphs."""
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                if len(pdf.pages) == 0:
                    return {"path": "TEXT", "confidence": 0.9, "sample_preview": "Empty PDF", "details": "PDF has no pages."}
                    
                # Check first 2 pages
                pages_to_check = min(2, len(pdf.pages))
                
                tables_found = 0
                text_extracted = ""
                combined_data = []
                
                for i in range(pages_to_check):
                    page = pdf.pages[i]
                    # Look for tables first
                    tables = page.extract_tables()
                    if tables:
                        tables_found += len(tables)
                        for table in tables:
                            combined_data.extend(table)
                    
                    # Extract text for fallback/preview
                    page_text = page.extract_text()
                    if page_text:
                        text_extracted += page_text + "\n"
                        
                sample_text = text_extracted[:200].replace('\n', ' ')
                
                if tables_found > 0 and len(combined_data) > 1:
                    # Load table into pandas (limit to 100 rows to match spec)
                    df = pd.DataFrame(combined_data[1:101], columns=combined_data[0])
                    
                    # Coerce columns to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    total_cols = len(df.columns)
                    
                    if total_cols > 0:
                        numeric_ratio = len(numeric_cols) / total_cols
                        is_data = numeric_ratio >= 0.5
                        
                        if is_data:
                            return {
                                "path": "DATA", 
                                "confidence": round(numeric_ratio, 2), 
                                "sample_preview": "PDF Data Table structure detected...", 
                                "details": f"Found {tables_found} tables. {len(numeric_cols)}/{total_cols} numeric columns."
                            }
                        else:
                            return {
                                "path": "TEXT",
                                "confidence": round(1 - numeric_ratio, 2),
                                "sample_preview": sample_text,
                                "details": f"PDF contains mixed table/text. Majority of columns non-numeric ({total_cols - len(numeric_cols)}/{total_cols})."
                            }
                
                # If no tables or weak tables, classify as TEXT
                return {
                    "path": "TEXT",
                    "confidence": 0.9,
                    "sample_preview": sample_text if sample_text else "Unstructured PDF Text...",
                    "details": f"PDF processed. Extracted {len(text_extracted)} characters across {pages_to_check} pages. No significant numeric tables found."
                }
        except Exception:
            return {
                "path": "TEXT",
                "confidence": 0.9,
                "sample_preview": "PDF Document",
                "details": "PDF format detected."
            }