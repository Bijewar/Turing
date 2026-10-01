"""
test_layer3_search_patents.py — Patents search (Serper.dev) service tests.

Catches every SWARM-5 BUG in services/layer3/search_patents.py:

  BUG #11 — no retry / exponential backoff for HTTP 429/500/timeout.
  Auth    — when SERPER_API_KEY is missing, the function must return an
            empty list with a clear warning (already implemented, this
            test guards against regression).

Design notes:
- Every test mocks httpx with `respx` so no real network call is made.
- Each test FAILS against the current buggy code (red) and PASSES only
  once the corresponding fix lands (green).
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from schemas.layer3 import StructuralQuery, SearchResult, SearchSource
from services.layer3 import search_patents as spt


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _query() -> StructuralQuery:
    return StructuralQuery(
        original_node_id="n0",
        original_confidence=50.0,
        structural_description="A high-variance input causes output failure.",
    )


def _patent(*, title="Test Patent", snippet="test snippet", link="https://patents.google.com/patent/US123",
            patentNumber="US123", assignee="Acme Corp", citationCount=5):
    return {
        "patents": [
            {
                "title": title,
                "snippet": snippet,
                "link": link,
                "patentNumber": patentNumber,
                "assignee": assignee,
                "citationCount": citationCount,
            }
        ]
    }


# ===========================================================================
# Rate limit / timeout handling (BUG #11)
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_handles_rate_limit(monkeypatch):
    """SWARM-5 BUG #11: HTTP 429 from Serper.dev must trigger retry with
    exponential backoff, not fail on the first attempt."""
    monkeypatch.setenv("SERPER_API_KEY", "test-key")

    route = respx.post(spt.SERPER_API_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json=_patent()),
        ]
    )
    sleep_mock = AsyncMock()

    with patch("asyncio.sleep", new=sleep_mock):
        results = await spt.search_patents(_query())

    assert route.call_count >= 2, (
        f"search_patents must retry on HTTP 429; only {route.call_count} attempt(s)."
    )
    assert sleep_mock.await_count >= 1, (
        "search_patents must sleep (exponential backoff) between 429 retries."
    )
    assert isinstance(results, list)


@pytest.mark.asyncio
@respx.mock
async def test_handles_timeout(monkeypatch):
    """SWARM-5 BUG #11: a network timeout must NOT crash the function.
    It must return an empty list (graceful degradation)."""
    monkeypatch.setenv("SERPER_API_KEY", "test-key")

    respx.post(spt.SERPER_API_URL).mock(
        side_effect=httpx.ConnectTimeout("timed out")
    )

    results = await spt.search_patents(_query())
    assert results == [], (
        "search_patents must return an empty list on timeout, not raise."
    )


# ===========================================================================
# Auth handling — missing API key
# ===========================================================================

@pytest.mark.asyncio
async def test_patent_api_auth_handled(monkeypatch):
    """When SERPER_API_KEY is missing, search_patents must return an empty
    list (graceful degradation), NOT crash with an opaque error.

    The current code already handles this (returns `results` early). This
    test guards against future regressions and verifies the behavior is
    graceful + logged.
    """
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    # Should NOT raise — must return [] gracefully.
    results = await spt.search_patents(_query())
    assert results == [], (
        f"search_patents must return [] when SERPER_API_KEY is missing; got {results}"
    )


@pytest.mark.asyncio
@respx.mock
async def test_patent_api_auth_handled_clear_error(monkeypatch, caplog):
    """SWARM-5 (auth clarity): when the patent API key is missing, the
    function must log a clear, actionable warning (so an operator can
    quickly diagnose the empty-results cause). The current code does
    `logger.warning("[search_patents] SERPER_API_KEY not set. ...")`
    which is good — this test guards against a regression that silently
    returns [] without warning.
    """
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    import logging
    with caplog.at_level(logging.WARNING, logger="services.layer3.search_patents"):
        results = await spt.search_patents(_query())

    assert results == []
    # A warning mentioning the missing key must be logged.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("SERPER_API_KEY" in r.getMessage() for r in warnings), (
        "search_patents must log a clear warning when SERPER_API_KEY is missing."
    )


# ===========================================================================
# Empty results
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_handles_empty_results(monkeypatch):
    """When the API returns zero patents, the function must return an
    empty list, not crash."""
    monkeypatch.setenv("SERPER_API_KEY", "test-key")

    respx.post(spt.SERPER_API_URL).mock(
        return_value=httpx.Response(200, json={"patents": []})
    )

    results = await spt.search_patents(_query())
    assert results == [], (
        f"empty Serper patents response must yield empty list, got {results}"
    )