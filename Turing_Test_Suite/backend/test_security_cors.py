"""
Security tests — CORS configuration.

Catches:
- SWARM-6 BUG #1 / SWARM-8 BUG #1 — `allow_origins=["*"]` combined with
  `allow_credentials=True` is an invalid and dangerous combination.
  Starlette's CORSMiddleware "fixes" the spec violation by REFLECTING
  the request Origin back in `Access-Control-Allow-Origin`, so any
  website can make credentialed requests to the API.
- SWARM-8 BUG #1 — origins must come from an env-var allowlist with a
  safe default.
- SWARM-8 BUG #1 — methods and headers must be restricted to the actual
  set used by the API, not `["*"]`.
- SWARM-8 BUG #1 — Origin-reflection attack must be blocked.
"""
import importlib
import inspect
import os
import re
import sys

import pytest


# ===========================================================================
# Source-level checks (deterministic, fast, fake-fix resistant).
# ===========================================================================

def _main_module_source() -> str:
    """Import `main` fresh and return its source code."""
    # Drop any cached copy so we always see the on-disk version.
    sys.modules.pop("main", None)
    import main as _m
    return inspect.getsource(_m)


def test_cors_not_wildcard_with_credentials():
    """SWARM-6 BUG #1: `allow_origins=["*"]` MUST NOT be combined with
    `allow_credentials=True`.

    The current code (main.py:44-50) does exactly this. The fix
    replaces `["*"]` with an explicit allowlist of trusted origins.

    Fake-fix resistance: this test inspects the literal source code so
    a `os.getenv("CORS_ORIGINS", "*")` default of `"*"` still fails.
    """
    src = _main_module_source()

    # Find the CORSMiddleware block.
    cors_match = re.search(
        r"add_middleware\(\s*CORSMiddleware.*?\)",
        src,
        re.DOTALL,
    )
    assert cors_match, "CORSMiddleware must be configured in main.py"
    cors_block = cors_match.group(0)

    # Reject any literal "*" or '["*"]' in the CORS block.
    assert not re.search(r'allow_origins\s*=\s*\[?\s*"\*"\s*\]?', cors_block), (
        "allow_origins must NOT be ['*'] — that combination with "
        "allow_credentials=True lets any origin make credentialed requests."
    )

    # And the credentials flag must be present (not absent by accident).
    assert re.search(r"allow_credentials\s*=\s*True", cors_block), (
        "allow_credentials=True must remain so trusted origins can "
        "send cookies; the fix is to restrict origins, not drop credentials."
    )


def test_cors_origins_from_env_var():
    """SWARM-8 BUG #1: CORS origins MUST be loaded from an env var with a
    safe default for dev.

    The current code hardcodes `["*"]`. The fix reads from an env var
    (e.g., `CORS_ORIGINS` or `ALLOWED_ORIGINS`) so production deployments
    can restrict the allowlist without code changes.
    """
    src = _main_module_source()

    cors_match = re.search(
        r"add_middleware\(\s*CORSMiddleware.*?\)",
        src,
        re.DOTALL,
    )
    assert cors_match, "CORSMiddleware must be configured in main.py"
    cors_block = cors_match.group(0)

    # Look for env-var-driven origin resolution.
    has_env_var = bool(re.search(
        r"os\.getenv\s*\(\s*[\"'](?:CORS_ORIGINS|ALLOWED_ORIGINS|ALLOWED_ORIGINS)[\"']",
        src,
    )) or bool(re.search(
        r"getenv\s*\(\s*[\"'](?:CORS_ORIGINS|ALLOWED_ORIGINS)[\"']",
        src,
    ))
    assert has_env_var, (
        "CORS origins must be loaded from an env var "
        "(CORS_ORIGINS / ALLOWED_ORIGINS), not hardcoded."
    )

    # And the env-var default (if any) must NOT be "*".
    default_match = re.search(
        r"os\.getenv\s*\(\s*[\"'](?:CORS_ORIGINS|ALLOWED_ORIGINS)[\"']"
        r"\s*,\s*[\"']([^\"']*)[\"']",
        src,
    )
    if default_match:
        default_val = default_match.group(1)
        assert "*" not in default_val, (
            f"CORS env var default must not contain '*'; got {default_val!r}"
        )


def test_cors_methods_restricted():
    """SWARM-8 BUG #1: `allow_methods=["*"]` MUST be replaced with the
    actual methods used (GET, POST, DELETE, OPTIONS, PATCH).

    The current code uses `["*"]` which expands to every HTTP method
    including ones the API never uses (CONNECT, TRACE). The fix lists
    only what's needed.
    """
    src = _main_module_source()
    cors_match = re.search(
        r"add_middleware\(\s*CORSMiddleware.*?\)",
        src,
        re.DOTALL,
    )
    assert cors_match, "CORSMiddleware must be configured in main.py"
    cors_block = cors_match.group(0)

    assert not re.search(r'allow_methods\s*=\s*\[?\s*"\*"\s*\]?', cors_block), (
        "allow_methods must NOT be ['*'] — list only the methods the API uses."
    )


def test_cors_headers_restricted():
    """SWARM-8 BUG #1: `allow_headers=["*"]` MUST be replaced with the
    actual headers used (Content-Type, Authorization, etc.)."""
    src = _main_module_source()
    cors_match = re.search(
        r"add_middleware\(\s*CORSMiddleware.*?\)",
        src,
        re.DOTALL,
    )
    assert cors_match, "CORSMiddleware must be configured in main.py"
    cors_block = cors_match.group(0)

    assert not re.search(r'allow_headers\s*=\s*\[?\s*"\*"\s*\]?', cors_block), (
        "allow_headers must NOT be ['*'] — list only the headers the API uses."
    )


# ===========================================================================
# Runtime checks via TestClient (also fake-fix resistant).
# ===========================================================================

def test_origin_reflection_attack_blocked(sync_client):
    """SWARM-8 BUG #1: a request with `Origin: https://evil.example`
    MUST NOT have that origin reflected back in
    `Access-Control-Allow-Origin`.

    The current code (allow_origins=["*"] + allow_credentials=True)
    makes Starlette's preflight handler reflect the request Origin.

    Fake-fix: any code that merely *says* `["*"]` is bad without fixing
    the credentials combo still fails this runtime test.
    """
    # Preflight (OPTIONS) request — the path Starlette reflects origins
    # on when allow_credentials=True + allow_origins=["*"].
    r = sync_client.options(
        "/api/runs/",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    acao = r.headers.get("access-control-allow-origin", "")
    assert acao != "https://evil.example", (
        f"CORS reflected the attacker origin {acao!r} — origin reflection "
        "attack succeeded. The fix must use an explicit allowlist and "
        "reject origins that are not on it."
    )


def test_cors_credentials_only_for_trusted_origins(sync_client):
    """SWARM-8 BUG #1: `Access-Control-Allow-Credentials: true` MUST
    NOT be returned when the request Origin is not on the allowlist.

    The current code returns credentials=true for every origin
    (combined with the reflected Origin header).
    """
    r = sync_client.options(
        "/api/runs/",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    acao = r.headers.get("access-control-allow-origin", "")
    acac = r.headers.get("access-control-allow-credentials", "")

    # Either the origin is rejected (no acao / acao is empty / acao is not
    # the evil origin) OR credentials are not sent.
    origin_allowed = acao == "https://evil.example"
    creds_sent = acac.lower() == "true"
    assert not (origin_allowed and creds_sent), (
        "CORS returned credentials=true for an untrusted origin — "
        "the allowlist fix is missing or wrong."
    )


def test_cors_wildcard_with_credentials_rejected_runtime(sync_client):
    """Runtime check: response to an evil origin MUST NOT contain the
    invalid `Access-Control-Allow-Origin: *` +
    `Access-Control-Allow-Credentials: true` combo (the literal current
    Starlette behaviour)."""
    r = sync_client.get("/health", headers={"Origin": "https://evil.example"})
    acao = r.headers.get("access-control-allow-origin", "")
    acac = r.headers.get("access-control-allow-credentials", "")
    # The CORS spec forbids this combo; browsers reject it.
    not_wildcard_plus_creds = not (
        acao == "*" and acac.lower() == "true"
    )
    assert not_wildcard_plus_creds, (
        "CORS returned `Access-Control-Allow-Origin: *` AND "
        "`Access-Control-Allow-Credentials: true` — invalid per CORS spec."
    )


def test_cors_trusted_origin_allowed(sync_client):
    """If the allowlist has a trusted origin, requests from that origin
    MUST be allowed (positive test, otherwise the fix could just
    reject all origins and pass the negative tests).

    We can't know the allowlist ahead of time, so we read it from the
    same env var the fix should use.
    """
    src = _main_module_source()
    env_match = re.search(
        r"os\.getenv\s*\(\s*[\"'](CORS_ORIGINS|ALLOWED_ORIGINS)[\"']"
        r"\s*,\s*[\"']([^\"']*)[\"']",
        src,
    )
    if not env_match:
        pytest.skip("Fix not yet implemented — no env var to read allowlist from.")

    var_name, default_val = env_match.group(1), env_match.group(2)
    raw = os.environ.get(var_name, default_val)
    if not raw or raw.strip() == "":
        pytest.skip(f"{var_name} empty — no trusted origin to test against.")

    trusted = raw.split(",")[0].strip()
    r = sync_client.options(
        "/api/runs/",
        headers={
            "Origin": trusted,
            "Access-Control-Request-Method": "POST",
        },
    )
    acao = r.headers.get("access-control-allow-origin", "")
    assert acao == trusted, (
        f"Trusted origin {trusted!r} was rejected ({acao!r}); the fix is "
        "supposed to allow it."
    )