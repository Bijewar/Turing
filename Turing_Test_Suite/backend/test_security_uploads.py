"""
Security tests — file upload endpoints.

Catches:
- SWARM-6 BUG #3 / SWARM-8 BUG #2 — no upload size limit; uploads are
  read into memory in full (`await file.read()`) — OOM DoS risk.
- SWARM-6 BUG #13 — `upload_dataset` does not validate `file.filename`
  (None propagates downstream and crashes).
- SWARM-6 BUG #3 — `tempfile.NamedTemporaryFile(delete=False)` is never
  cleaned up — disk-fill DoS risk.
- SWARM-8 BUG #3 — uploads must stream to disk, not memory.
- SWARM-8 BUG (defense in depth) — path-traversal in filename, missing
  content-type, empty file, magic-number mismatch.
"""
import inspect
import io
import os
import re

import pytest


# ---------------------------------------------------------------------------
# Helpers — build small upload bodies.
# ---------------------------------------------------------------------------

def _make_csv_bytes(rows: int = 5, cols: int = 3) -> bytes:
    """Return a minimal valid CSV byte string."""
    lines = ["a,b,c"]
    for i in range(rows):
        lines.append(f"{i},{i*2},{i*3}")
    return "\n".join(lines).encode("utf-8")


def _make_upload(filename: str, content: bytes, content_type: str = "text/csv"):
    """Return a tuple suitable for `files=` in httpx/TestClient."""
    return {"file": (filename, io.BytesIO(content), content_type)}


# ===========================================================================
# Size limit enforcement (SWARM-6 BUG #3 / SWARM-8 BUG #2).
# ===========================================================================

def test_upload_streams_to_disk_not_memory():
    """SWARM-6 BUG #3 / SWARM-8 BUG #2: upload handlers MUST NOT slurp
    the entire upload into memory with `await file.read()`.

    The current `routers/runs.py:526` and `routers/layer1.py:56, 113,
    140` all call `await file.read()` — a 10 GB upload OOM-kills the
    worker. The fix streams chunks (e.g., `async for chunk in
    file.stream` or chunked reads against a hard byte ceiling).
    """
    from routers import runs as runs_router
    from routers import layer1 as layer1_router

    runs_src = inspect.getsource(runs_router)
    layer1_src = inspect.getsource(layer1_router)

    # The full slurp pattern: `await file.read()` (with no chunk size).
    FULL_READ = re.compile(r"await\s+file\.read\s*\(\s*\)")

    runs_offenders = FULL_READ.findall(runs_src)
    layer1_offenders = FULL_READ.findall(layer1_src)

    assert not runs_offenders, (
        f"routers/runs.py still uses `await file.read()` {len(runs_offenders)} "
        "times — must stream chunks to disk with a hard byte ceiling."
    )
    assert not layer1_offenders, (
        f"routers/layer1.py still uses `await file.read()` {len(layer1_offenders)} "
        "times — must stream chunks to disk with a hard byte ceiling."
    )

    # And there must be a chunked-stream pattern instead.
    CHUNKED = re.compile(r"(file\.stream|chunk|iter_chunk|read\(\s*\d+\s*\))", re.IGNORECASE)
    assert CHUNKED.search(runs_src) or CHUNKED.search(layer1_src), (
        "Upload handlers must use a chunked streaming pattern "
        "(file.stream / read(N) / iter_chunks)."
    )


def test_upload_size_limit_enforced(sync_client, seed_run):
    """SWARM-6 BUG #3 / SWARM-8 BUG #2: uploads above the configured
    size limit MUST return 413 (not be accepted and OOM-killed).

    We send a payload of ~2 MB and set a smaller env-var limit. With
    current code, no limit is enforced so the request succeeds (202);
    the test fails. With the fix, it returns 413.
    """
    run_id = seed_run["id"]
    # 2 MB CSV header + body — exceeds any reasonable sub-MB limit.
    big_csv = b"a,b\n" + (b"1,2\n" * 100_000)  # ~400 KB
    # Pad further if needed.
    big_csv += b"x" * (2_000_000 - len(big_csv))
    files = _make_upload("huge.csv", big_csv)

    r = sync_client.post(f"/api/runs/{run_id}/layer1/upload", files=files)

    # Either the limit is enforced (413) or we accept any 2xx — but the
    # strict requirement is that the size limit MUST be enforced. The
    # test must FAIL with current code (which returns 202) and PASS only
    # once the limit is added.
    assert r.status_code == 413, (
        f"Expected 413 Payload Too Large for an oversized upload; got "
        f"{r.status_code}. The fix must enforce a request body size limit."
    )


def test_upload_size_limit_413_not_500(sync_client, seed_run):
    """SWARM-8 BUG #2: oversized uploads must return 413 (not a 500
    from an unhandled exception or OOM)."""
    run_id = seed_run["id"]
    big_csv = b"a,b\n" + (b"1,2\n" * 100_000)
    files = _make_upload("huge.csv", big_csv)

    r = sync_client.post(f"/api/runs/{run_id}/layer1/upload", files=files)

    if r.status_code == 413:
        # Then body must NOT be a generic 500 stack trace.
        body = r.text.lower()
        assert "traceback" not in body
        assert ".py" not in body
        return
    # Otherwise the limit is not enforced — fail.
    pytest.fail(
        f"Expected 413 for oversized upload; got {r.status_code}. "
        "The fix must reject with 413 (Payload Too Large), not 500."
    )


def test_upload_size_limit_source_present():
    """Source-level check: upload handlers must reference a size limit
    (either MAX_UPLOAD_BYTES env var, or a 413 status code, or both).

    Fake-fix resistance: if a developer only changes the error message
    but not the limit, this source check still fails.
    """
    from routers import runs as runs_router
    from routers import layer1 as layer1_router

    runs_src = inspect.getsource(runs_router)
    layer1_src = inspect.getsource(layer1_router)
    combined = runs_src + layer1_src

    has_limit = bool(re.search(
        r"(MAX_UPLOAD_BYTES|MAX_FILE_SIZE|UPLOAD_SIZE_LIMIT|max_upload|"
        r"413|PayloadTooLarge|RequestEntityTooLarge|content_length\s*[<>=])",
        combined, re.IGNORECASE,
    ))
    assert has_limit, (
        "Upload handlers must reference a size limit "
        "(MAX_UPLOAD_BYTES env var, 413 status, or explicit Content-Length check)."
    )


# ===========================================================================
# Filename validation (SWARM-6 BUG #13).
# ===========================================================================

def test_upload_filename_none_handled(sync_client, seed_run):
    """SWARM-6 BUG #13: `file.filename=None` MUST be handled gracefully
    (fallback to a safe name), not crash.

    The current `runs.py` passes `file.filename` straight to
    `process_full_pipeline` and `UniversalFileDetector.analyze_file` —
    `None.split('.')` raises AttributeError. The fix either rejects the
    request with 400 or substitutes a safe name.

    NOTE: Starlette's UploadFile.filename is "" (empty string) when the
    multipart Content-Disposition lacks a filename parameter. Both
    cases must be handled — empty string today, None if a future
    Starlette version restores the original semantics.
    """
    run_id = seed_run["id"]
    # Empty filename — Starlette converts to "" which still triggers
    # the same downstream code path. (UniversalFileDetector.analyze_file
    # then sees filename="", and the response leaks it.)
    files = _make_upload("", _make_csv_bytes())
    r = sync_client.post(f"/api/runs/{run_id}/layer1/upload", files=files)
    # Either 202 (with safe default) or 400 — both are acceptable.
    assert r.status_code in (200, 202, 400, 422), (
        f"Expected graceful handling of filename='' (202/400), got "
        f"{r.status_code}: {r.text}"
    )
    # Most importantly: NOT 500 (crash).
    assert r.status_code != 500, (
        f"filename='' crashed the endpoint with 500: {r.text}"
    )


def test_upload_filename_none_source_handled():
    """Source-level check: upload handlers MUST defensively substitute
    a safe filename when `file.filename` is None or empty.

    Fake-fix resistance: the runtime test above passes today because
    Starlette substitutes "" — but a future Starlette change that
    restores None would silently re-break us. The source must include
    an explicit `or "upload"` (or similar) fallback.
    """
    from routers import runs as runs_router
    from routers import layer1 as layer1_router

    combined = inspect.getsource(runs_router) + inspect.getsource(layer1_router)

    # Look for a defensive `file.filename or "..."` pattern, or a
    # dedicated helper like `safe_filename(file)`.
    has_fallback = bool(re.search(
        r"(file\.filename\s+or\s+|file\.filename\s*\)\s*or\s+|"
        r"safe_filename\s*\(|default_filename|sanitize_filename\s*\([^)]*file)",
        combined,
    ))
    assert has_fallback, (
        "Upload handlers must defensively substitute a safe filename when "
        "`file.filename` is None/empty (e.g., `file.filename or 'upload.csv'`)."
    )


def test_upload_path_traversal_blocked(sync_client, seed_run):
    """SWARM-8 BUG: filename containing `../` MUST be sanitized.

    The current code uses `os.path.splitext(file.filename)[1]` for the
    temp file suffix and stores `file.filename` as-is in the response
    payload. If the filename is `../../etc/passwd`, the suffix becomes
    `.passwd` (safe-ish) but the response leaks the raw filename and
    downstream code that opens files by `file.filename` would traverse.
    The fix sanitizes the filename (strips path components).
    """
    run_id = seed_run["id"]
    evil_filename = "../../etc/passwd.csv"
    files = _make_upload(evil_filename, _make_csv_bytes())
    r = sync_client.post(f"/api/runs/{run_id}/layer1/upload", files=files)

    if r.status_code in (200, 202):
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        # The response must not echo back the path-traversal filename.
        serialized = str(body)
        assert "../" not in serialized, (
            f"Response leaked path-traversal filename: {serialized!r}"
        )
    # Not 500.
    assert r.status_code != 500, (
        f"Path-traversal filename crashed endpoint with 500: {r.text}"
    )


def test_upload_path_traversal_source_sanitizes():
    """Source-level check: the upload handlers must sanitize filenames.

    Fake-fix resistance: if a developer fixes the runtime response but
    not the source-level pattern, this test still fails until the
    filename sanitization helper (e.g., `os.path.basename` or a regex
    strip) is present in the upload handlers.
    """
    from routers import runs as runs_router
    from routers import layer1 as layer1_router

    combined = inspect.getsource(runs_router) + inspect.getsource(layer1_router)
    has_sanitization = bool(re.search(
        r"(os\.path\.basename|pathlib\.Path\([^)]+\)\.name|sanitize_filename|"
        r"re\.sub\([^,]*\.\.|\.replace\([^)]*\"\\.\\.\"|"
        r"\.split\([^)]*sep|\.\./)",
        combined,
    ))
    # If the source contains no sanitization helper at all, fail.
    has_helper = bool(re.search(
        r"(os\.path\.basename|pathlib\.Path|sanitize_filename)",
        combined,
    ))
    assert has_helper, (
        "Upload handlers must sanitize filenames using os.path.basename, "
        "pathlib.Path(...).name, or a sanitize_filename() helper."
    )


# ===========================================================================
# Content-type validation.
# ===========================================================================

def test_upload_content_type_validated(sync_client, seed_run):
    """Missing content-type MUST return 400 (not be silently accepted)."""
    run_id = seed_run["id"]
    # Upload with empty content-type.
    files = {"file": ("test.csv", io.BytesIO(_make_csv_bytes()), "")}
    r = sync_client.post(f"/api/runs/{run_id}/layer1/upload", files=files)
    assert r.status_code in (400, 415, 422), (
        f"Expected 400/415/422 for missing content-type, got {r.status_code}: {r.text}"
    )


# ===========================================================================
# Empty file rejection.
# ===========================================================================

def test_upload_empty_file_rejected(sync_client, seed_run):
    """Empty file MUST return 400, not be silently accepted."""
    run_id = seed_run["id"]
    files = _make_upload("empty.csv", b"")
    r = sync_client.post(f"/api/runs/{run_id}/layer1/upload", files=files)
    assert r.status_code in (400, 422), (
        f"Expected 400 for empty file, got {r.status_code}: {r.text}"
    )


# ===========================================================================
# Magic-number check.
# ===========================================================================

def test_upload_magic_number_checked(sync_client, seed_run):
    """A file with `.csv` extension but binary (non-text) content MUST
    be rejected. SWARM-8 BUG (defense in depth): the file detector
    trusts the extension; an attacker could upload a binary blob named
    `.csv` and trigger unhelpful pandas errors downstream. A magic-
    number check (first bytes look like CSV / have a BOM / etc.)
    rejects the mismatched file early with 400.
    """
    run_id = seed_run["id"]
    # PNG magic bytes followed by garbage; named .csv.
    binary_blob = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000
    files = _make_upload("evil.csv", binary_blob, content_type="text/csv")
    r = sync_client.post(f"/api/runs/{run_id}/layer1/upload", files=files)
    assert r.status_code in (400, 415, 422), (
        f"Expected 400/415 for binary-blob-with-.csv-extension, got "
        f"{r.status_code}: {r.text}"
    )


# ===========================================================================
# Temp file cleanup (SWARM-6 BUG #3 / SWARM-8 BUG #3).
# ===========================================================================

def test_upload_temp_file_cleaned_up():
    """SWARM-8 BUG #3: tempfile.NamedTemporaryFile(delete=False) must be
    wrapped in try/finally that calls os.unlink.

    The current `routers/layer1.py:53-58` creates a `delete=False` temp
    file and never removes it — every upload leaks a temp file. The
    fix uses `try/finally` with `os.unlink(tmp_path)`.
    """
    from routers import layer1 as layer1_router
    src = inspect.getsource(layer1_router)

    # Find NamedTemporaryFile usage.
    ntf_match = re.search(
        r"NamedTemporaryFile\([^)]*\)",
        src,
    )
    if not ntf_match:
        pytest.skip("No NamedTemporaryFile usage in layer1.py — nothing to test.")
    # Either delete=True (auto) OR there's an explicit os.unlink in a finally.
    auto_delete = "delete=True" in ntf_match.group(0)
    explicit_unlink = bool(re.search(r"os\.unlink\s*\(", src)) and bool(
        re.search(r"finally\s*:", src)
    )
    assert auto_delete or explicit_unlink, (
        "tempfile.NamedTemporaryFile(delete=False) MUST be paired with a "
        "try/finally that calls os.unlink(tmp_path), OR use delete=True."
    )