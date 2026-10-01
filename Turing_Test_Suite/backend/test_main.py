r"""
test_main.py — FastAPI app setup tests for main.py.

Catches every SWARM-6 BUG in main.py / app configuration:
  BUG #1  — CORS `allow_origins=["*"]` with `allow_credentials=True`
  BUG #2  — No global exception handler — stack traces leak on 500
  BUG #3  — No request body size limit
  BUG #4  — No rate limiting on any endpoint
  BUG #5  — Lifespan never disposes the DB engine on shutdown
  BUG #6  — /health always returns 200 — does not verify DB reachability

Design notes:
- Tests use a mix of: import-and-inspect (for middleware registration),
  httpx.AsyncClient (for endpoint behaviour), and direct lifespan
  invocation (for startup/shutdown).
- No real LLM call is made.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _app_middlewares(app) -> list:
    """Return the list of middleware wrappers registered on the FastAPI app."""
    # FastAPI stores middleware as `user_middleware` (list of MiddlewareFactory)
    return list(getattr(app, "user_middleware", []))


def _middleware_classes(app) -> list:
    """Return the class of each registered middleware."""
    return [m.cls for m in _app_middlewares(app)]


def test_cors_not_wildcard_with_credentials(app):
    """SWARM-6 BUG #1: `allow_origins=["*"]` MUST NOT be combined with
    `allow_credentials=True`. The CORS spec (and FastAPI's CORSMiddleware)
    would reflect any Origin back, letting any website make credentialed
    requests. The fixed code must use an explicit allowlist."""
    from fastapi.middleware.cors import CORSMiddleware

    cors_mws = [m for m in _app_middlewares(app) if m.cls is CORSMiddleware]
    assert cors_mws, "CORSMiddleware must be registered."
    cors = cors_mws[0]
    kwargs = cors.kwargs

    allow_origins = kwargs.get("allow_origins", [])
    allow_credentials = kwargs.get("allow_credentials", False)

    # If allow_credentials is True, allow_origins MUST NOT be ["*"].
    if allow_credentials:
        assert allow_origins != ["*"], (
            f"allow_origins=['*'] + allow_credentials=True is a CORS misconfiguration. "
            f"Got allow_origins={allow_origins!r}, allow_credentials={allow_credentials!r}"
        )
        assert "*" not in allow_origins, (
            f"Wildcard origin '*' in allow_origins with allow_credentials=True is "
            f"the bug. Got allow_origins={allow_origins!r}"
        )


def test_cors_origins_from_env(app, monkeypatch):
    """SWARM-6 BUG #1 (env-driven): CORS origins must be configurable via
    an env var (e.g. `CORS_ALLOWED_ORIGINS` or `ALLOWED_ORIGINS`) with a
    sensible dev default — not hardcoded to ["*"]."""
    from fastapi.middleware.cors import CORSMiddleware

    cors_mws = [m for m in _app_middlewares(app) if m.cls is CORSMiddleware]
    assert cors_mws, "CORSMiddleware must be registered."
    kwargs = cors_mws[0].kwargs
    allow_origins = kwargs.get("allow_origins", [])

    # The origins must NOT be a hardcoded wildcard.
    assert allow_origins != ["*"], (
        f"CORS allow_origins is hardcoded to ['*'] — must come from env var. "
        f"Got {allow_origins!r}"
    )
    # The origins must be a list (possibly with a sensible dev default).
    assert isinstance(allow_origins, list), (
        f"CORS allow_origins must be a list; got {type(allow_origins).__name__}"
    )


def test_global_exception_handler_registered(app):
    """SWARM-6 BUG #2: A global exception handler for `Exception` must be
    registered so 500s don't leak stack traces. The handler returns a
    generic error message and logs the actual exception server-side.

    FastAPI always registers handlers for `HTTPException` and
    `RequestValidationError` — those are NOT a global handler. The handler
    key must be `Exception` itself (the base class), so any uncaught
    exception is intercepted."""
    # FastAPI exposes exception handlers via `app.exception_handlers`.
    handlers = getattr(app, "exception_handlers", {})
    handler_keys = list(handlers.keys())
    # The base `Exception` class itself must be a key.
    has_global = any(key is Exception for key in handler_keys)
    assert has_global, (
        "No global exception handler keyed on `Exception` is registered — 500s "
        f"leak stack traces. Registered handler keys: {handler_keys!r}"
    )


def test_request_id_middleware_present(app):
    """SWARM-6 BUG (request tracing): Every response should carry an
    `X-Request-Id` header so requests can be traced across logs. Verify
    that a request-id middleware is registered."""
    # Check that some middleware class whose name suggests request-id is
    # registered, OR that the middleware stack sets X-Request-Id on
    # responses.
    mws = _middleware_classes(app)
    names = [getattr(c, "__name__", str(c)) for c in mws]

    has_request_id = any(
        "request" in n.lower() and "id" in n.lower() for n in names
    )
    if not has_request_id:
        # Fall back: check whether responses actually include X-Request-Id.
        # The middleware might be a custom class with an unrelated name.
        from starlette.testclient import TestClient
        with TestClient(app) as client:
            r = client.get("/health")
            has_request_id = "X-Request-Id" in r.headers or "x-request-id" in {k.lower() for k in r.headers}
    assert has_request_id, (
        f"No request-id middleware registered. Middleware names: {names!r}"
    )


def test_request_size_limit_middleware_present(app):
    """SWARM-6 BUG #3: A request body size limit middleware must be
    registered so upload endpoints don't accept arbitrarily large files.
    Look for any middleware that checks Content-Length / request body size."""
    mws = _middleware_classes(app)
    names = [getattr(c, "__name__", str(c)) for c in mws]

    # Look for size-related middleware by name OR by source code inspection.
    has_size_limit = any(
        "size" in n.lower() or "limit" in n.lower() or "upload" in n.lower()
        for n in names
    )
    if not has_size_limit:
        # Inspect source code of any custom middleware classes for size-limit
        # logic.
        for cls in mws:
            try:
                src = inspect.getsource(cls)
            except (OSError, TypeError):
                continue
            if "content_length" in src.lower() or "max_upload" in src.lower() or "request_size" in src.lower():
                has_size_limit = True
                break
    assert has_size_limit, (
        f"No request-size-limit middleware registered. Middleware names: {names!r}"
    )


def test_rate_limiting_middleware_present(app):
    """SWARM-6 BUG #4: Rate limiting must be configured for expensive
    endpoints (upload, layer3/search, layer4/report) so a single client
    can't drive unlimited LLM spend. Look for any rate-limit middleware."""
    mws = _middleware_classes(app)
    names = [getattr(c, "__name__", str(c)) for c in mws]

    has_rate_limit = any(
        "rate" in n.lower() or "throttle" in n.lower() or "limit" in n.lower()
        for n in names
    )
    if not has_rate_limit:
        # Inspect source code for rate-limit logic.
        for cls in mws:
            try:
                src = inspect.getsource(cls)
            except (OSError, TypeError):
                continue
            if "rate_limit" in src.lower() or "ratelimit" in src.lower() or "throttle" in src.lower():
                has_rate_limit = True
                break
    assert has_rate_limit, (
        f"No rate-limit middleware registered. Middleware names: {names!r}"
    )


@pytest.mark.asyncio
async def test_lifespan_disposes_engine_on_shutdown(app, monkeypatch):
    """SWARM-6 BUG #5: The lifespan context manager must call
    `await engine.dispose()` on shutdown. The buggy code only calls
    `init_db()` on startup and never disposes the engine, leaking
    connections."""
    from database import database as db_module
    from sqlalchemy.ext.asyncio import AsyncEngine

    dispose_calls = {"n": 0}
    original_dispose = AsyncEngine.dispose

    async def _record_dispose(self, *args, **kwargs):
        dispose_calls["n"] += 1
        # Don't actually dispose — we don't want to break the shared in-memory
        # engine used by other tests.
        return None

    # Patch at the class level — AsyncEngine instances use __slots__ and
    # don't allow per-instance attribute assignment.
    monkeypatch.setattr(AsyncEngine, "dispose", _record_dispose)

    async with app.router.lifespan_context(app):
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
    # At this point shutdown has fired.

    assert dispose_calls["n"] >= 1, (
        "engine.dispose() was never called on shutdown — DB connections leak."
    )


@pytest.mark.asyncio
async def test_health_check_verifies_db(app, monkeypatch):
    """SWARM-6 BUG #6: `/health` must actually verify DB reachability
    (e.g. run `SELECT 1` against the engine). The buggy code always
    returns 200 / `{"status":"healthy"}` even when the DB is down.

    We simulate DB-down by patching the engine's `connect()` to raise,
    then verify /health returns 503 (or non-200) and an unhealthy status."""
    from database import database as db_module
    from sqlalchemy.ext.asyncio import AsyncEngine
    from httpx import AsyncClient, ASGITransport

    class _DBDown(Exception):
        pass

    # Patch at the class level — AsyncEngine instances use __slots__.
    async def _raise_on_connect(self, *args, **kwargs):
        raise _DBDown("DB is unreachable")

    monkeypatch.setattr(AsyncEngine, "connect", _raise_on_connect)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")

    # The health check MUST NOT return 200 with `{"status":"healthy"}` when
    # the DB is unreachable. It should return 503 or a non-healthy status.
    assert resp.status_code != 200 or resp.json().get("status") != "healthy", (
        f"/health returned {resp.status_code} with {resp.json()!r} even though "
        "the DB is down — the health check must verify DB reachability."
    )


def test_security_headers_middleware_present(app):
    """SWARM-6 BUG (security headers): The app must add security headers
    to every response:
      X-Content-Type-Options: nosniff
      X-Frame-Options: DENY
      Content-Security-Policy: ...
      Strict-Transport-Security: ...
    The buggy code adds no security headers middleware."""
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/health")

    required_headers = {
        "x-content-type-options": "nosniff",
        "x-frame-options": "deny",
    }
    optional_headers = [
        "content-security-policy",
        "strict-transport-security",
    ]

    lower_headers = {k.lower(): v for k, v in r.headers.items()}

    missing_required = []
    for header, expected_value in required_headers.items():
        if header not in lower_headers:
            missing_required.append(header)
        elif lower_headers[header].lower() != expected_value:
            missing_required.append(f"{header} (got {lower_headers[header]!r})")

    missing_optional = [h for h in optional_headers if h not in lower_headers]

    # Required headers MUST be present.
    assert not missing_required, (
        f"Missing required security headers: {missing_required!r}. "
        f"Present headers: {lower_headers!r}"
    )
    # At least one of CSP / HSTS should be present.
    assert not missing_optional, (
        f"Missing recommended security headers: {missing_optional!r}. "
        f"Present headers: {lower_headers!r}"
    )


@pytest.mark.asyncio
async def test_app_has_health_endpoint(async_client):
    """/health must exist and return 200."""
    resp = await async_client.get("/health")
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_app_has_root_endpoint(async_client):
    """/ must exist and return something sensible (status: ok, service
    info, or a redirect)."""
    resp = await async_client.get("/")
    assert resp.status_code in (200, 301, 302, 307, 308), (
        f"Root endpoint returned unexpected status {resp.status_code}"
    )


def test_app_includes_all_routers(app):
    """SWARM-6 BUG (router completeness): All 5 routers must be included:
    runs, layer1, layer2, layer3, layer4. A missing router means a whole
    layer of functionality is unreachable."""
    paths = {route.path for route in app.routes}

    expected_prefixes = [
        "/api/runs",
        "/api/layer1",
        "/api/layer2",
        "/api/layer3",
        "/api/layer4",
    ]
    missing = [p for p in expected_prefixes if not any(
        path.startswith(p) for path in paths
    )]
    assert not missing, (
        f"Missing routers for prefixes: {missing!r}. "
        f"All paths: {sorted(paths)!r}"
    )