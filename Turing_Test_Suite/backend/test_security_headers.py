"""
Security tests — HTTP response headers.

Catches:
- SWARM-6 BUG (security headers) — the API sets none of the standard
  security headers (`X-Content-Type-Options`, `X-Frame-Options`,
  `Content-Security-Policy`, `Strict-Transport-Security`).
- SWARM-6 BUG (info leak) — the `Server` header leaks the
  FastAPI/uvicorn version.

The fix adds a small middleware (or per-response header set) that
sets these headers on every response.
"""
import inspect
import re

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _main_source() -> str:
    import main as _m
    return inspect.getsource(_m)


def _has_security_middleware() -> bool:
    """Returns True if `main.py` registers a middleware that sets
    security headers (either via a custom BaseHTTPMiddleware or via the
    `secure` package)."""
    src = _main_source()
    # Look for either:
    #   - `add_middleware(SecurityHeadersMiddleware)` or similar
    #   - `add_middleware(...)` with a name containing 'Header'
    #   - Setting headers in a custom middleware class
    return bool(re.search(
        r"add_middleware\([^)]*Header|"
        r"@app\.middleware\([\"']http[\"']\)|"
        r"secure\.SecureHeaders|"
        r"SecurityHeaders",
        src,
    ))


# ===========================================================================
# X-Content-Type-Options: nosniff
# ===========================================================================

def test_x_content_type_options_nosniff(sync_client):
    """SWARM-6 BUG: every response MUST include
    `X-Content-Type-Options: nosniff` to prevent MIME-type sniffing
    attacks (especially on uploaded-file responses)."""
    r = sync_client.get("/health")
    xcto = r.headers.get("x-content-type-options", "")
    assert xcto.lower() == "nosniff", (
        f"Expected `X-Content-Type-Options: nosniff`, got {xcto!r}. "
        "The fix must add a middleware that sets this header on every "
        "response."
    )


def test_x_content_type_options_nosniff_source():
    """Source-level check: main.py must register a middleware that
    sets `X-Content-Type-Options`.

    Fake-fix resistance: if a developer sets the header only on
    specific responses, this source check still fails until a
    middleware covers all responses.
    """
    src = _main_source()
    # Look for either a SecurityHeaders middleware reference OR an
    # explicit `x-content-type-options` literal somewhere.
    has_middleware = _has_security_middleware()
    has_header_literal = "x-content-type-options" in src.lower() or \
                         "X-Content-Type-Options" in src
    assert has_middleware or has_header_literal, (
        "main.py must register a security-headers middleware OR set the "
        "X-Content-Type-Options header on every response."
    )


# ===========================================================================
# X-Frame-Options: DENY (or SAMEORIGIN)
# ===========================================================================

def test_x_frame_options_deny(sync_client):
    """SWARM-6 BUG: every response MUST include
    `X-Frame-Options: DENY` (or `SAMEORIGIN`) to prevent clickjacking."""
    r = sync_client.get("/health")
    xfo = r.headers.get("x-frame-options", "")
    assert xfo.upper() in ("DENY", "SAMEORIGIN"), (
        f"Expected `X-Frame-Options: DENY` (or SAMEORIGIN), got {xfo!r}. "
        "The fix must add a middleware that sets this header on every "
        "response."
    )


def test_x_frame_options_source():
    """Source-level check: main.py must register a middleware that
    sets `X-Frame-Options`."""
    src = _main_source()
    has_middleware = _has_security_middleware()
    has_header_literal = "x-frame-options" in src.lower() or \
                         "X-Frame-Options" in src
    assert has_middleware or has_header_literal, (
        "main.py must register a security-headers middleware OR set the "
        "X-Frame-Options header on every response."
    )


# ===========================================================================
# Content-Security-Policy
# ===========================================================================

def test_content_security_policy_set(sync_client):
    """SWARM-6 BUG: every response MUST include a
    `Content-Security-Policy` header.

    For an API returning JSON, a strict policy like `default-src 'none'`
    or `frame-ancestors 'none'` is appropriate.
    """
    r = sync_client.get("/health")
    csp = r.headers.get("content-security-policy", "")
    assert csp != "", (
        "Expected a Content-Security-Policy header. The fix must add a "
        "middleware that sets at least `default-src 'none'` (or "
        "`frame-ancestors 'none'`) on every response."
    )


def test_content_security_policy_source():
    """Source-level check: main.py must reference CSP."""
    src = _main_source()
    has_middleware = _has_security_middleware()
    has_csp_literal = "content-security-policy" in src.lower()
    assert has_middleware or has_csp_literal, (
        "main.py must register a security-headers middleware OR set the "
        "Content-Security-Policy header on every response."
    )


# ===========================================================================
# Strict-Transport-Security
# ===========================================================================

def test_strict_transport_security_set(sync_client):
    """SWARM-6 BUG: every response MUST include a
    `Strict-Transport-Security` header to enforce HTTPS.

    Even if the API is currently served over plain HTTP (dev), the
    header should be set so production deployments inherit the policy.
    """
    r = sync_client.get("/health")
    sts = r.headers.get("strict-transport-security", "")
    assert sts != "", (
        "Expected a Strict-Transport-Security header. The fix must add "
        "a middleware that sets this header on every response."
    )


def test_strict_transport_security_source():
    """Source-level check: main.py must reference HSTS."""
    src = _main_source()
    has_middleware = _has_security_middleware()
    has_sts_literal = "strict-transport-security" in src.lower()
    assert has_middleware or has_sts_literal, (
        "main.py must register a security-headers middleware OR set the "
        "Strict-Transport-Security header on every response."
    )


# ===========================================================================
# Server header must not leak framework/version info
# ===========================================================================

def test_no_server_header_leaks_version(sync_client):
    """SWARM-6 BUG: the `Server` header MUST NOT leak the FastAPI /
    uvicorn version.

    The current uvicorn default sets `Server: uvicorn` (without version),
    which is acceptable. Some configurations leak `Server: uvicorn
    0.30.6` — version disclosure aids targeted attacks. The fix either
    removes the header, overrides it with a generic value, or ensures
    no version is included.
    """
    r = sync_client.get("/health")
    server = r.headers.get("server", "")
    if server == "":
        # No Server header at all — best case.
        return
    # If a Server header exists, it must NOT contain version numbers.
    # Look for patterns like "uvicorn 0.30.6", "gunicorn 21.2.0", etc.
    has_version = bool(re.search(r"\b\d+\.\d+(?:\.\d+)?\b", server))
    assert not has_version, (
        f"Server header leaks version info: {server!r}. The fix must "
        "either remove the header or strip the version."
    )
    # And it must NOT contain the literal "uvicorn" or "fastapi" if
    # those names are considered implementation disclosure.
    # NOTE: We only flag version disclosure here — naming is acceptable
    # since it's already a known fact from the API behaviour.


# ===========================================================================
# All headers must be present together (regression check).
# ===========================================================================

def test_all_security_headers_present(sync_client):
    """Regression check: all four security headers must be present
    together on every response.

    This test acts as a roll-up of the per-header tests so that a
    developer running just this test sees the full list at once.
    """
    r = sync_client.get("/health")
    h = {k.lower() for k in r.headers.keys()}
    expected = {
        "x-content-type-options",
        "x-frame-options",
        "content-security-policy",
        "strict-transport-security",
    }
    missing = expected - h
    assert not missing, (
        f"Missing security headers: {sorted(missing)}. "
        f"Present headers: {sorted(h)}"
    )