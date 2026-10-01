"""
test_routers_layer4.py

Tests for `python-engine/routers/layer4.py`.

Catches SWARM-6 BUGs #29, #30 + SWARM-6 BUG #23 (error leakage).
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

import os

ROUTER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "python-engine", "routers", "layer4.py")


def _read_source():
    with open(ROUTER_PATH) as f:
        return f.read()


# ===========================================================================
# run_id is UUID-validated (BUG #30)
# ===========================================================================
class TestRunIdValidation:
    """Locks in SWARM-6 BUG #30: generate_report takes `run_id` as a query
    parameter with no UUID validation and no DB lookup. Any string is
    accepted, generating an LLM report (costing tokens) for non-existent runs."""

    def test_run_id_is_path_param_not_query(self):
        """Verify run_id is a PATH parameter (`/report/{run_id}`), not a
        query parameter (`/report?run_id=...`)."""
        source = _read_source()
        # Look for `@router.post("/report/{run_id}")` (path) vs
        # `@router.post("/report")` (then run_id is query).
        if "@router.post(\"/report\")" in source or "@router.post('/report')" in source:
            # Path is `/report` — run_id is a query param. That's the bug.
            pytest.fail(
                "/report endpoint uses `/report` (no path param) — run_id "
                "is a query param. Make it `/report/{run_id}` so it's a path "
                "param, validated by the framework."
            )

    def test_run_id_validated_as_uuid(self):
        """Verify run_id is validated as a UUID (either via the route type
        annotation `run_id: UUID` or explicit `uuid.UUID(run_id)` check)."""
        source = _read_source()
        has_uuid_import = "import uuid" in source or "from uuid" in source
        has_uuid_check = "uuid.UUID(run_id)" in source or "run_id: UUID" in source or "run_id: uuid" in source
        if not (has_uuid_import and has_uuid_check):
            pytest.fail(
                "/report endpoint doesn't validate run_id as a UUID. Add "
                "`uuid.UUID(run_id)` validation or change the type annotation "
                "to `run_id: UUID`."
            )

    def test_run_id_db_lookup_returns_404_for_missing(self):
        """Verify the endpoint looks up the run in the DB and returns 404
        if missing."""
        source = _read_source()
        # Look for `crud.get_run` or `await db.execute(select(...))`.
        if "crud.get_run" not in source and "select(" not in source:
            pytest.fail(
                "/report endpoint doesn't look up the run in the DB. Calls "
                "for non-existent runs succeed silently, wasting LLM tokens. "
                "Add `run = await crud.get_run(db, run_id); if not run: "
                "raise HTTPException(404)`."
            )


# ===========================================================================
# Idempotent (don't regenerate same report twice)
# ===========================================================================
class TestIdempotency:
    def test_report_endpoint_is_idempotent(self):
        """Same run_id + bridges twice should return the same report (not
        regenerate, wasting tokens)."""
        source = _read_source()
        # Look for any cache or DB-persist check before generating.
        # If the endpoint always calls `build_report`, it's not idempotent.
        cache_indicators = [
            "if run.report", "if run.layer4", "if existing_report",
            "cached_report", "get_cached_report",
        ]
        has_cache = any(ind in source for ind in cache_indicators)
        if not has_cache:
            # Check if the endpoint at least persists the report (so future
            # calls could read it).
            if "crud.update" not in source and "run.report" not in source:
                pytest.fail(
                    "/report endpoint is not idempotent — same run_id + "
                    "bridges twice regenerates the report, wasting LLM tokens. "
                    "Cache the generated report or persist it to the DB and "
                    "return the cached version on subsequent calls."
                )


# ===========================================================================
# Internal exceptions not leaked (BUG #29)
# ===========================================================================
class TestNoInternalExceptionLeak:
    def test_no_str_exc_in_http_exception(self):
        source = _read_source()
        bad_patterns = [
            r"HTTPException\([^)]*detail\s*=\s*str\(exc\)",
            r"HTTPException\([^)]*detail\s*=\s*str\(e\)",
            r"HTTPException\([^)]*detail\s*=\s*str\(err\)",
        ]
        for pat in bad_patterns:
            if re.search(pat, source, re.DOTALL):
                pytest.fail(
                    "Router uses `HTTPException(..., detail=str(exc))` — "
                    "internal LLM error messages (including API key substrings) "
                    "leak to the client."
                )


# ===========================================================================
# Sanitize API keys from error responses (BUG #23)
# ===========================================================================
class TestSanitizeErrorMessages:
    """Locks in SWARM-6 BUG #23: `build_report` failure path sets
    `theMechanism = f"Failed to generate report: {e}"`. The raw exception
    message is exposed to the client, which may include API key substrings
    like `sk-ant-xxx`."""

    def test_error_message_sanitized(self):
        """Verify error messages returned to the client don't contain
        raw exception text (which may have API keys)."""
        source = _read_source()
        # Look for `f"Failed to generate report: {e}"` or similar.
        if re.search(r"f\"[^\"]*\{e\}\"", source) or re.search(r"f'[^']*\{e\}'", source):
            pytest.fail(
                "Router interpolates raw `{e}` into the error response. "
                "Sanitize: redact API keys (sk-*, sk-ant-*), file paths, and "
                "internal module names before returning to the client."
            )


# ===========================================================================
# Response model set
# ===========================================================================
class TestResponseModelSet:
    def test_endpoint_has_response_model(self):
        source = _read_source()
        idx = source.find("async def generate_report")
        if idx == -1:
            pytest.skip("generate_report endpoint not found.")
        decorator_start = source.rfind("@router.", 0, idx)
        decorator_end = source.find(")\n", decorator_start)
        decorator = source[decorator_start:decorator_end]
        if "response_model" not in decorator:
            pytest.fail(
                "generate_report has no `response_model=` — internal fields "
                "may leak to the client."
            )


# ===========================================================================
# Streaming endpoint — disconnect handling (if applicable)
# ===========================================================================
class TestStreamingDisconnect:
    """If there's a stream endpoint, verify it handles client disconnect."""

    def test_stream_endpoint_handles_disconnect(self):
        """Verify the stream endpoint (if any) uses `request: Request`
        and calls `await request.is_disconnected()`."""
        source = _read_source()
        if "stream" not in source.lower():
            pytest.skip("No stream endpoint in layer4 router.")
        # Look for `request: Request` in the endpoint signature.
        if "request: Request" not in source:
            pytest.fail(
                "Stream endpoint doesn't take `request: Request` — it can't "
                "detect client disconnect, leading to a leaked generator "
                "that polls RUNS_STORE forever after the client closes."
            )
        # Also look for `is_disconnected`.
        if "is_disconnected" not in source:
            pytest.fail(
                "Stream endpoint takes `request: Request` but doesn't call "
                "`await request.is_disconnected()` in the loop. The generator "
                "runs forever after the client disconnects."
            )