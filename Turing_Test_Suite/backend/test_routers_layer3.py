"""
test_routers_layer3.py

Tests for `python-engine/routers/layer3.py`.

Catches SWARM-5 BUGs #31, #32 and SWARM-6 BUG #29.
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import patch, MagicMock

import pytest

import os

ROUTER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "python-engine", "routers", "layer3.py")


def _read_source():
    with open(ROUTER_PATH) as f:
        return f.read()


# ===========================================================================
# Concurrency semaphore on external API calls (BUG #31)
# ===========================================================================
class TestConcurrencySemaphore:
    """Locks in SWARM-5 BUG #31: no concurrency semaphore on the /search
    endpoint — all 4 external API calls (Wikipedia, web, papers, patents)
    fire concurrently without a cap. Under load, this trips rate limits
    on every external API simultaneously."""

    def test_search_endpoint_has_semaphore(self):
        """Verify the search endpoint uses `asyncio.Semaphore` (or similar)
        to cap concurrent external API calls."""
        source = _read_source()
        # Find the search endpoint.
        m = re.search(r"async\s+def\s+search\s*\([^)]*\).*?(?=\nasync\s+def\s|\Z)",
                      source, re.DOTALL)
        if not m:
            pytest.skip("/search endpoint not found")
        body = m.group(0)
        # Look for asyncio.Semaphore or similar.
        if "Semaphore" not in body and "semaphore" not in body.lower():
            pytest.fail(
                "/search endpoint has no concurrency semaphore — all 4 "
                "external API calls fire concurrently. Add `asyncio.Semaphore(N)` "
                "around each external API call to cap concurrency (e.g., N=5)."
            )


# ===========================================================================
# Caching of search results (BUG #32)
# ===========================================================================
class TestSearchResultCaching:
    """Locks in SWARM-5 BUG #32: no caching of search results — same query
    twice calls all 4 external APIs twice, doubling cost and rate-limit
    pressure."""

    def test_search_endpoint_caches_results(self):
        """Verify the search endpoint caches results by query hash."""
        source = _read_source()
        # Look for cache infrastructure: functools.lru_cache, cachetools,
        # Redis, a module-level dict cache, or a TTL cache.
        cache_indicators = [
            "lru_cache", "TTLCache", "cached", "@cache", "functools.cache",
            "RedisCache", "aioredis", "cache.set", "cache.get",
            "_search_cache", "_result_cache",
        ]
        has_cache = any(ind in source for ind in cache_indicators)
        if not has_cache:
            pytest.fail(
                "/search endpoint has no caching — same query twice calls "
                "all 4 external APIs twice. Add a cache (lru_cache, cachetools "
                "TTLCache, or Redis) keyed on the query hash."
            )


# ===========================================================================
# Query length validation
# ===========================================================================
class TestQueryValidation:
    def test_search_validates_query_not_empty(self):
        """Empty query should return 422 (Pydantic validation) or 400,
        not call all external APIs with empty query."""
        source = _read_source()
        # Look for a Pydantic model with min_length or empty-check on query.
        # If the query field is just `str` with no constraints, it's a bug.
        m = re.search(r"query\s*:\s*str\s*=", source)
        if m:
            # Default is set but no constraint — look for Field(min_length=1).
            if "min_length" not in source and "min_length=1" not in source:
                pytest.fail(
                    "/search query field has no min_length constraint — "
                    "empty query is accepted and wastes external API calls."
                )

    def test_search_validates_query_max_length(self):
        """Very long query (>1000 chars) should be rejected."""
        source = _read_source()
        if "max_length" not in source:
            pytest.fail(
                "/search query has no max_length — a 1MB query string is "
                "accepted, blowing the LLM context window and external "
                "API URL length limits."
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
                    "internal exception messages (including external API "
                    "key substrings) leak to the client."
                )


# ===========================================================================
# Response model set
# ===========================================================================
class TestResponseModelSet:
    @pytest.mark.parametrize("endpoint_name", ["search", "search_stream"])
    def test_endpoint_has_response_model(self, endpoint_name):
        source = _read_source()
        idx = source.find(f"async def {endpoint_name}")
        if idx == -1:
            pytest.skip(f"Endpoint {endpoint_name} not found.")
        decorator_start = source.rfind("@router.", 0, idx)
        decorator_end = source.find(")\n", decorator_start)
        decorator = source[decorator_start:decorator_end]
        if "response_model" not in decorator:
            # For SSE endpoints, response_model may be StreamingResponse.
            if "StreamingResponse" in decorator or "media_type" in decorator:
                pass  # SSE endpoints don't have response_model — OK.
            else:
                pytest.fail(
                    f"Endpoint `{endpoint_name}` has no `response_model=` "
                    "— internal fields may leak to the client."
                )


# ===========================================================================
# Broad exception swallowing (SWARM-5)
# ===========================================================================
class TestNoBroadExceptionSwallow:
    """SWARM-5: routers catch `Exception` and swallow it. A specific error
    (e.g., 429 from Wikipedia) becomes a generic 500."""

    def test_no_bare_except(self):
        source = _read_source()
        # Look for `except:` (bare) or `except Exception:` (broad).
        bare_except = re.findall(r"^\s*except\s*:", source, re.MULTILINE)
        broad_except = re.findall(r"^\s*except\s+Exception\s*:", source, re.MULTILINE)
        # Bare `except:` is always bad.
        if bare_except:
            pytest.fail(
                f"Found {len(bare_except)} bare `except:` statements — these "
                "swallow ALL exceptions including KeyboardInterrupt. Catch "
                "specific exception types."
            )