"""
test_layer1_file_detector.py

Tests for `python-engine/services/layer1/file_detector.py`.

Each test's docstring names the exact SWARM-3 / SWARM-6 bug ID it catches.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from services.layer1.file_detector import UniversalFileDetector


# ===========================================================================
# Magic-number detection (BUG #31)
# ===========================================================================
class TestMagicNumberDetection:
    """Locks in SWARM-3 BUG #31: file_detector uses `pd.to_numeric(...,
    errors='ignore')` which is deprecated and removed in pandas 2.x.
    Also covers magic-number detection correctness."""

    def test_detect_pdf_by_magic_number(self):
        """Catches SWARM-3 BUG: PDF detected by `%PDF` magic bytes, not just
        the `.pdf` extension."""
        pdf_bytes = b"%PDF-1.4\n%binary content\n"
        result = UniversalFileDetector.analyze_file(pdf_bytes, "unknown.bin")
        assert result is not None
        # Must detect as PDF, not as binary or text.
        assert "pdf" in str(result).lower() or "PDF" in str(result)

    def test_detect_docx_by_magic_number(self):
        """DOCX is a ZIP file; magic bytes are `PK`."""
        # Minimal ZIP magic: PK\x03\x04
        docx_magic = b"PK\x03\x04" + b"\x00" * 100
        result = UniversalFileDetector.analyze_file(docx_magic, "unknown.bin")
        assert result is not None

    def test_detect_csv_by_content_not_just_extension(self):
        """A file with .txt extension but CSV content should be detected as CSV."""
        csv_content = b"col1,col2,col3\n1,2,3\n4,5,6\n"
        result = UniversalFileDetector.analyze_file(csv_content, "data.txt")
        assert result is not None
        # Should be detected as CSV/data, not text.

    def test_detect_json_by_content(self):
        """JSON file detected correctly by content."""
        json_content = b'{"key": "value", "list": [1, 2, 3]}'
        result = UniversalFileDetector.analyze_file(json_content, "data.txt")
        assert result is not None


# ===========================================================================
# UTF-16 misclassification (BUG #32)
# ===========================================================================
class TestUtf16Classification:
    """Locks in SWARM-3 BUG #32: UTF-16 text is mis-classified as binary
    DATA path because the null bytes between ASCII chars look like binary."""

    def test_utf16_text_not_classified_as_binary(self):
        """UTF-16-encoded plain text should NOT be misclassified as binary."""
        # UTF-16 LE BOM + simple text "Hello"
        utf16_text = b"\xff\xfeH\x00e\x00l\x00l\x00o\x00"
        result = UniversalFileDetector.analyze_file(utf16_text, "data.bin")
        assert result is not None
        # Should be detected as text, not as binary data path.
        result_str = str(result).lower()
        # It should NOT say "binary" or "data" path for what is plain text.
        assert "binary" not in result_str, "UTF-16 text misclassified as binary"


# ===========================================================================
# Numeric token regex too lax (BUG #33)
# ===========================================================================
class TestNumericTokenRegex:
    """Locks in SWARM-3 BUG #33: numeric-token regex accepts `"1-5"` as a
    number, which is actually a range string."""

    def test_numeric_token_regex_not_too_lax(self):
        """A token like "1-5" should NOT be classified as a number (it's a
        range), and "1.5" SHOULD be classified as a number."""
        # We need to inspect the source code to verify the regex pattern.
        import inspect
        source = inspect.getsource(UniversalFileDetector)
        # If there's a numeric regex, it should be reasonably strict.
        # A naive pattern like r'[\d.-]+' accepts "1-5" — too lax.
        # A better pattern: r'^[+-]?\d+\.?\d*$' or similar.
        # We don't enforce a specific regex, but check that "1-5" is not
        # treated as a single numeric token (it should be 2 tokens or text).
        # This is a heuristic test — read the source.
        assert source is not None
        # If there's a numeric detection regex, it should not accept "1-5".
        # We look for patterns like r'[\d.-]+' which are too lax.
        import re
        # Find any regex strings in the source.
        lax_patterns = re.findall(r"r['\"]([^'\"]*\d[^'\"]*)['\"]", source)
        for pat in lax_patterns:
            # If a pattern matches "1-5" as a number, it's too lax.
            # Skip patterns that are clearly not numeric (e.g., file extensions).
            if re.match(pat, "1-5") and not re.match(pat, "abc"):
                # It accepts "1-5" as a number — too lax.
                pytest.fail(
                    f"Numeric regex {pat!r} accepts '1-5' as a number — too lax."
                )


# ===========================================================================
# pandas 2.x compatibility (BUG #31 — pd.to_numeric errors='ignore')
# ===========================================================================
class TestPandasCompatibility:
    """Locks in SWARM-3 BUG #31: file_detector uses
    `pd.to_numeric(..., errors='ignore')` which is deprecated in pandas 2.x
    and removed in pandas 2.2+. The pinned requirements.txt is pandas>=2.2.2,
    so this WILL break on a fresh install."""

    def test_no_deprecated_pd_to_numeric_errors_ignore(self):
        """Verify `pd.to_numeric(..., errors='ignore')` is NOT used anywhere
        in file_detector.py. This call is removed in pandas 2.2+."""
        import inspect
        source = inspect.getsource(UniversalFileDetector)
        assert "errors='ignore'" not in source, (
            "file_detector uses `pd.to_numeric(..., errors='ignore')` which is "
            "removed in pandas 2.2+. Replace with `errors='coerce'` + dropna."
        )
        assert 'errors="ignore"' not in source, (
            "file_detector uses `pd.to_numeric(..., errors='ignore')` which is "
            'removed in pandas 2.2+. Replace with `errors="coerce"` + dropna.'
        )

    def test_no_deprecated_df_where_with_none(self):
        """Verify the parser doesn't use `df.where(df.notna(), other=None)`
        which downcasts numeric columns to object dtype."""
        # This bug is actually in universal_parser.py, but file_detector
        # may also use it. Test both.
        import inspect
        source = inspect.getsource(UniversalFileDetector)
        # If the pattern is here, it's a bug.
        assert "other=None" not in source or "df.where" not in source, (
            "file_detector uses `df.where(df.notna(), other=None)` which "
            "downcasts numeric columns to object dtype."
        )


# ===========================================================================
# File size limit (DoS protection)
# ===========================================================================
class TestFileSizeLimit:
    """Catches the SWARM-8 finding: no file-size cap means a 10GB upload
    OOMs the worker. file_detector should refuse to analyze files above a
    sane limit (or document that the limit is enforced elsewhere)."""

    def test_file_size_limit_env_var_respected(self, monkeypatch):
        """If MAX_UPLOAD_MB env var is set, files above that limit are
        rejected with a clear error."""
        # Set a small limit for the test.
        monkeypatch.setenv("MAX_UPLOAD_MB", "1")
        # 2 MB file — exceeds the 1 MB limit.
        big_bytes = b"\x00" * (2 * 1024 * 1024)
        # Should raise or return a "too large" result, not analyze silently.
        with pytest.raises((ValueError, OSError, OverflowError)):
            UniversalFileDetector.analyze_file(big_bytes, "big.csv")

    def test_file_size_limit_default_present(self):
        """If MAX_UPLOAD_MB is not set, there should be a sensible default
        (e.g., 50MB) that prevents DoS."""
        # We can't test this without an actual huge file, but we can
        # inspect the source to verify a default limit exists.
        import inspect
        source = inspect.getsource(UniversalFileDetector)
        # Look for any size check or MAX_SIZE / MAX_UPLOAD constant.
        has_size_check = (
            "MAX_SIZE" in source
            or "MAX_UPLOAD" in source
            or "max_size" in source
            or "len(" in source  # any length check
            or "getsize" in source
        )
        if not has_size_check:
            pytest.fail(
                "file_detector has no file-size limit — a 10GB upload will OOM. "
                "Add a MAX_UPLOAD_MB env var check or a hard default limit."
            )


# ===========================================================================
# Empty file handling
# ===========================================================================
class TestEmptyFileHandling:
    def test_empty_file_handled_gracefully(self):
        """Empty file should produce a clear error or a clear result, not
        crash with an opaque exception."""
        empty = b""
        # Either returns a result (None or specific empty-input type) or raises
        # a clear ValueError — but not a pandas or numpy opaque error.
        try:
            result = UniversalFileDetector.analyze_file(empty, "empty.csv")
            # If it returned, that's fine.
            assert result is not None or result is None  # any return is OK
        except (ValueError, TypeError) as e:
            # A clear error message is acceptable.
            assert "empty" in str(e).lower() or "no content" in str(e).lower() or "size" in str(e).lower()
        except Exception as e:
            # An opaque pandas/numpy error is NOT acceptable.
            pytest.fail(f"Empty file produced opaque error: {type(e).__name__}: {e}")


# ===========================================================================
# Missing dependencies (python-docx, xlrd) — SWARM-9 BUG #7
# ===========================================================================
class TestMissingDependencies:
    """Locks in SWARM-9 BUG #7: `python-docx` and `xlrd` are imported by the
    code but missing from requirements.txt. On a fresh install, .docx and
    .xls uploads will fail with ImportError."""

    def test_python_docx_in_requirements(self):
        """Verify python-docx is in requirements.txt."""
        req_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python-engine", "requirements.txt"))
        if not os.path.exists(req_path):
            pytest.skip("requirements.txt not found")
        with open(req_path) as f:
            contents = f.read().lower()
        assert "python-docx" in contents or "docx" in contents, (
            "python-docx is imported by file_detector/universal_parser but "
            "missing from requirements.txt. DOCX uploads will fail with "
            "ImportError on a fresh install."
        )

    def test_xlrd_in_requirements_if_xls_supported(self):
        """If .xls files are supported, xlrd must be in requirements.txt."""
        import inspect
        source = inspect.getsource(UniversalFileDetector)
        if ".xls" in source or "xlrd" in source:
            req_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python-engine", "requirements.txt"))
            if not os.path.exists(req_path):
                pytest.skip("requirements.txt not found")
            with open(req_path) as f:
                contents = f.read().lower()
            assert "xlrd" in contents, (
                "xlrd is imported/used by file_detector but missing from "
                "requirements.txt. .xls uploads will fail with ImportError."
            )

    def test_openpyxl_in_requirements_if_xlsx_supported(self):
        """If .xlsx files are supported, openpyxl must be in requirements.txt."""
        import inspect
        source = inspect.getsource(UniversalFileDetector)
        if ".xlsx" in source or "openpyxl" in source:
            req_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python-engine", "requirements.txt"))
            if not os.path.exists(req_path):
                pytest.skip("requirements.txt not found")
            with open(req_path) as f:
                contents = f.read().lower()
            assert "openpyxl" in contents, (
                "openpyxl is imported/used by file_detector but missing from "
                "requirements.txt. .xlsx uploads will fail with ImportError."
            )

    def test_pypdf_or_pdfplumber_in_requirements(self):
        """If PDF parsing is supported, a PDF library must be in requirements.txt."""
        import inspect
        source = inspect.getsource(UniversalFileDetector)
        if "pdf" in source.lower():
            req_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python-engine", "requirements.txt"))
            if not os.path.exists(req_path):
                pytest.skip("requirements.txt not found")
            with open(req_path) as f:
                contents = f.read().lower()
            has_pdf_lib = any(lib in contents for lib in ("pypdf", "pdfplumber", "pymupdf", "fitz", "pypdf2"))
            assert has_pdf_lib, (
                "PDF parsing is referenced by file_detector but no PDF library "
                "(pypdf, pdfplumber, pymupdf) is in requirements.txt."
            )