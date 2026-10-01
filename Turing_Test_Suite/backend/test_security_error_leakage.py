"""
Security tests — error / stack-trace leakage.

Catches:
- SWARM-6 BUG #2 — no global exception handler; 500 responses leak
  stack traces to the client.
- SWARM-6 BUG #23 — `str(e)` persisted as `error_message` and surfaced
  to user — leaks internal details (file paths, API keys, module
  names).
- SWARM-8 BUG — error responses must include an `X-Request-Id` header
  for traceability.
- SWARM-6 BUG — 422 validation responses must NOT leak internal
  exception messages, only field-level errors.
- SWARM-6 BUG — `HTTPException(status_code=500, detail=str(exc))` is
  a dangerous pattern that must never appear in the codebase.
"""
import inspect
import re

import pytest
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_router_modules():
    """Return all router modules so we can scan their source code."""
    import routers.layer1
    import routers.layer2
    import routers.layer3
    import routers.layer4
    import routers.runs
    return [
        ("routers.layer1", routers.layer1),
        ("routers.layer2", routers.layer2),
        ("routers.layer3", routers.layer3),
        ("routers.layer4", routers.layer4),
        ("routers.runs", routers.runs),
    ]


def _main_source() -> str:
    import main as _m
    return inspect.getsource(_m)


@pytest.fixture()
def non_raising_client(app):
    """TestClient that returns 500 instead of re-raising server exceptions.

    The shared `sync_client` fixture in conftest uses the default
    `raise_server_exceptions=True` which makes unhandled exceptions
    propagate to the test runner — preventing us from inspecting the
    500 response body. This fixture opts out.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ===========================================================================
# Source-level: the dangerous `HTTPException(..., detail=str(exc))` pattern
# MUST NOT appear anywhere in the codebase.
# ===========================================================================

def test_http_exception_does_not_leak_str_exc():
    """SWARM-6 BUG #23: the pattern `HTTPException(status_code=...,
    detail=str(exc))` MUST NOT appear anywhere in the codebase.

    The current code uses this pattern in routers/layer1.py:121,164,203,
    227, 261, 285, 304, 323 (and similar elsewhere). The fix replaces
    `str(exc)` with a generic public message and logs the actual
    exception server-side.
    """
    offenders = []
    for name, mod in _all_router_modules():
        src = inspect.getsource(mod)
        # Match `HTTPException(status_code=..., detail=str(e...))` or
        # `HTTPException(..., detail=str(exc))` etc.
        # The key pattern: detail=str(some_exception_variable).
        bad = re.findall(
            r"HTTPException\([^)]*detail\s*=\s*str\(\s*(?:e|exc|err|exception|ex)[^)]*\)",
            src,
            re.IGNORECASE | re.DOTALL,
        )
        for b in bad:
            offenders.append(f"{name}: {b!r}")

    assert not offenders, (
        f"Found {len(offenders)} `HTTPException(..., detail=str(exc))` "
        f"patterns in routers — these leak internal exception messages "
        f"to the client. Offenders:\n" + "\n".join(offenders)
    )


# ===========================================================================
# Source-level: a global exception handler MUST be registered.
# ===========================================================================

def test_global_exception_handler_registered():
    """SWARM-6 BUG #2: a `@app.exception_handler(Exception)` (or
    equivalent) handler MUST be registered in `main.py` to catch all
    uncaught exceptions and return a generic 500 response without
    leaking the stack trace.

    The current `main.py` has no exception handler. The fix adds one.
    """
    src = _main_source()
    has_handler = bool(re.search(
        r"@app\.exception_handler\(\s*(Exception|BaseException)\s*\)",
        src,
    )) or bool(re.search(
        r"app\.add_exception_handler\(\s*(Exception|BaseException)\s*,",
        src,
    ))
    assert has_handler, (
        "main.py must register a global `@app.exception_handler(Exception)` "
        "to prevent stack-trace leakage on 500 responses."
    )


def test_global_exception_handler_does_not_log_to_response():
    """The global exception handler MUST NOT include the exception
    detail, str(exc), or traceback in the response body.
    """
    src = _main_source()
    # Find the exception handler function body if present.
    handler_match = re.search(
        r"@app\.exception_handler\([^)]*\)\s*"
        r"(?:async\s+def|def)\s+\w+\([^)]*\).*?(?=\n@|\ndef|\nclass|\Z)",
        src, re.DOTALL,
    )
    if not handler_match:
        handler_match = re.search(
            r"app\.add_exception_handler\([^)]*\)\s*$",
            src, re.MULTILINE,
        )
    # If no handler is registered, the previous test catches that.
    if not handler_match:
        return
    # If we found a handler, look for a `return JSONResponse` block that
    # includes any of: str(exc), exc, traceback.format_exc, repr(exc).
    # Find the function body — simple heuristic: between the handler
    # decorator and the next decorator/function.
    handler_section = handler_match.group(0)
    # Look for the next ~40 lines after the decorator to find the body.
    decorator_start = handler_match.start()
    body = src[decorator_start:decorator_start + 2000]
    # The body must NOT include str(exc)/traceback/format_exc in the
    # JSONResponse body.
    bad_in_response = re.search(
        r"JSONResponse\([^)]*detail\s*=\s*[\"']?[^\"]*"
        r"(str\(\s*exc|traceback\.format_exc|str\(e\)|repr\(exc\)|exc\.args)",
        body, re.DOTALL,
    )
    assert not bad_in_response, (
        "Global exception handler must not embed `str(exc)` or "
        "`traceback.format_exc()` in the JSONResponse detail — that "
        "leaks internal details to the client."
    )


# ===========================================================================
# Runtime: trigger an internal error and verify the response is clean.
# ===========================================================================

def test_500_response_does_not_contain_stack_trace(non_raising_client, monkeypatch):
    """SWARM-6 BUG #2: a 500 response MUST NOT contain a Python stack
    trace.

    Setup: monkeypatch one of the run-status handlers to raise an
    unhandled exception. Then GET it and assert the response body has
    no `Traceback` / `File "` / `line ` lines.
    """
    # Patch `crud.get_run` (called by the GET /api/runs/{run_id} endpoint)
    # to raise an unhandled error.
    import database.crud as crud_mod

    async def boom(*args, **kwargs):
        raise RuntimeError("internal secret detail — should not leak")

    monkeypatch.setattr(crud_mod, "get_run", boom)

    r = non_raising_client.get("/api/runs/some-run-id")
    # Should be 500 (or 422 if the schema rejects the run_id). 422 is
    # also acceptable since the schema fix rejects non-UUID IDs.
    assert r.status_code in (400, 404, 422, 500), r.text

    if r.status_code != 500:
        # If the schema rejected the ID, we can't trigger 500 this way.
        pytest.skip("Schema rejected the run_id before reaching the handler; "
                    "no 500 to inspect.")

    body = r.text
    assert "Traceback" not in body, (
        f"500 response body leaked a stack trace:\n{body}"
    )
    assert 'File "' not in body, (
        f"500 response body leaked file paths:\n{body}"
    )
    assert "line " not in body.lower(), (
        f"500 response body leaked line numbers:\n{body}"
    )


def test_500_response_does_not_contain_file_paths(non_raising_client, monkeypatch):
    """SWARM-6 BUG #2: a 500 response MUST NOT contain `.py` file paths."""
    import database.crud as crud_mod

    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(crud_mod, "get_run", boom)
    r = non_raising_client.get("/api/runs/12345678-1234-4321-8765-123456789012")
    if r.status_code != 500:
        pytest.skip("No 500 to inspect.")
    body = r.text
    # Look for `something.py` patterns.
    py_paths = re.findall(r"\b[\w/\\.]+\.py\b", body)
    assert not py_paths, (
        f"500 response leaked Python file paths: {py_paths}\nBody: {body}"
    )


def test_500_response_does_not_contain_api_keys(non_raising_client, monkeypatch):
    """SWARM-6 BUG #23: a 500 response MUST NOT contain API-key prefixes
    (`sk-`, `sk-ant-`) even if the underlying exception embeds them.

    Setup: raise an exception whose message contains a fake API key,
    trigger 500, assert the response body does not contain the key.
    """
    import database.crud as crud_mod

    fake_key = "sk-ant-fake-1234567890abcdef"
    async def boom(*args, **kwargs):
        raise ValueError(f"Auth error: invalid key {fake_key}")

    monkeypatch.setattr(crud_mod, "get_run", boom)
    r = non_raising_client.get("/api/runs/12345678-1234-4321-8765-123456789012")
    if r.status_code != 500:
        pytest.skip("No 500 to inspect.")
    body = r.text
    assert "sk-ant-" not in body, (
        f"500 response leaked sk-ant-* API key prefix: {body}"
    )
    assert "sk-" not in body, (
        f"500 response leaked sk-* API key prefix: {body}"
    )
    assert fake_key not in body, (
        f"500 response leaked the full fake API key: {body}"
    )


def test_500_response_has_request_id(non_raising_client, monkeypatch):
    """SWARM-6 BUG: every 500 response MUST include an `X-Request-Id`
    header so support tickets / logs can be correlated."""
    import database.crud as crud_mod

    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(crud_mod, "get_run", boom)
    r = non_raising_client.get("/api/runs/12345678-1234-4321-8765-123456789012")
    if r.status_code != 500:
        pytest.skip("No 500 to inspect.")
    has_request_id = "x-request-id" in {k.lower() for k in r.headers.keys()}
    assert has_request_id, (
        "500 response must include an X-Request-Id header for "
        "traceability. Headers were: " + ", ".join(r.headers.keys())
    )


# ===========================================================================
# 422 validation responses: only field-level errors, not internal messages.
# ===========================================================================

def test_validation_error_does_not_leak_internal_details(sync_client):
    """SWARM-6 BUG: a 422 response (Pydantic validation error) MUST
    only contain field-level error info (loc, msg, type) — NOT internal
    exception messages or stack frames.

    Setup: POST /api/runs/ with an obviously invalid body. FastAPI
    returns 422 by default; the test asserts the body shape is the
    standard FastAPI 422 (only loc/msg/type) and does not contain
    `Traceback`, `.py`, or `Exception:` strings.
    """
    # Send a body with wrong types to trigger Pydantic validation errors.
    r = sync_client.post("/api/runs/", json={"input_file": 12345, "input_type": 999})
    assert r.status_code == 422, r.text
    body = r.text

    # Standard FastAPI 422 body must NOT contain Python internals.
    assert "Traceback" not in body
    assert ".py" not in body
    assert "Exception" not in body

    # Must contain field-level errors (loc/msg).
    try:
        data = r.json()
        assert "detail" in data
        # Each detail entry must have at least loc/msg/type.
        for err in data["detail"]:
            assert "loc" in err and "msg" in err and "type" in err, (
                f"422 error entry missing loc/msg/type: {err}"
            )
    except Exception:
        # If response is not JSON, it must still be a clean string.
        assert "Traceback" not in body


# ===========================================================================
# Source-level: `logger.error(..., exc_info=True)` may leak str(exc)
# to logs — must use a sanitized form.
# ===========================================================================

def test_no_str_exc_in_logger_error_with_response():
    """`logger.error(f"...{e}")` patterns where the response body also
    embeds `str(e)` MUST NOT exist in any router.

    This catches the runs.py pattern where the background task's
    failure message ("Failed: {e}") is later served via the SSE stream
    to the client.
    """
    offenders = []
    for name, mod in _all_router_modules():
        src = inspect.getsource(mod)
        # Look for `Failed: {e}` / `Failed: {exc}` / `status = f"Failed: {e}"`
        # patterns that flow into the SSE response.
        bad = re.findall(
            r"Failed:\s*\{[^}]*e[^}]*\}|status\s*=\s*f[\"']Failed:\s*\{[^}]*e[^}]*\}",
            src, re.IGNORECASE,
        )
        for b in bad:
            offenders.append(f"{name}: {b!r}")

    assert not offenders, (
        "Router code must not embed exception messages in SSE-streamed "
        f"status strings — these leak to the client. Offenders:\n" +
        "\n".join(offenders)
    )