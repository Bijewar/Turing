"""
test_layer3_search_papers.py — Semantic Scholar (papers) search service tests.

Catches every SWARM-5 BUG in services/layer3/search_papers.py:

  BUG #11 — no retry / exponential backoff for HTTP 429/500/timeout.
  BUG #12 — Semantic Scholar call missing `User-Agent` header → 403 risk.
  BUG #13 — `citation_count` from Semantic Scholar not cast to int —
             TypeError when the API returns a string like `"42"`.
  BUG #14 — `replication_count` fabricated from `citation_count` via
             `max(0, (citation_count - 10) // 50)`. This is NOT a real
             replication count — it's a fabricated heuristic that lies about
             whether the result has been independently replicated.
  arxiv   — generic arXiv rate limit / timeout / malformed-XML handling.

Design notes:
- Every test mocks httpx with `respx` so no real network call is made.
- Each test FAILS against the current buggy code (red) and PASSES only
  once the corresponding fix lands (green).
"""
from __future__ import annotations

import asyncio
import inspect
import math
import os
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from schemas.layer3 import StructuralQuery, SearchResult, SearchSource
from services.layer3 import search_papers as sp


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _query() -> StructuralQuery:
    return StructuralQuery(
        original_node_id="n0",
        original_confidence=50.0,
        structural_description="A high-variance input causes output failure.",
    )


def _paper(
    *,
    title="Test Paper",
    abstract="Test abstract.",
    citationCount=42,
    year=2020,
    url="https://example.com/paper",
    externalIds=None,
):
    return {
        "title": title,
        "abstract": abstract,
        "citationCount": citationCount,
        "year": year,
        "url": url,
        "externalIds": externalIds or {"DOI": "10.1000/test"},
    }


def _s2_response(papers):
    return {"data": papers, "total": len(papers)}


# ===========================================================================
# BUG #12: missing User-Agent header
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_semantic_scholar_user_agent_set():
    """SWARM-5 BUG #12: the Semantic Scholar API requires a descriptive
    User-Agent header (their docs explicitly warn they 403 generic
    library UA strings like `python-httpx/x.y.z`).

    The other three search services set a custom UA
    (`CausalNexus/1.0 (research bot; ...)`); search_papers does NOT.
    """
    captured = {}

    def _capture(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=_s2_response([_paper()]))

    respx.get(sp.SEMANTIC_SCHOLAR_API).mock(side_effect=_capture)

    results = await sp.search_papers(_query())

    assert "user-agent" in captured["headers"], (
        f"search_papers must send a User-Agent header; got headers: {captured['headers']}"
    )
    ua = captured["headers"]["user-agent"]
    # Generic httpx UA is `python-httpx/x.y.z` — must NOT be used.
    assert not ua.startswith("python-httpx/"), (
        f"search_papers must set a custom User-Agent (not the generic `python-httpx/`); "
        f"got: {ua!r}"
    )
    # The custom UA must be descriptive (contain a project name / contact).
    assert len(ua) > 10, (
        f"User-Agent must be descriptive (e.g., `CausalNexus/1.0 (research bot)`); got: {ua!r}"
    )


# ===========================================================================
# BUG #13: citation_count not cast to int — TypeError on string
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_citation_count_parsed_safely():
    """SWARM-5 BUG #13: `citationCount` from Semantic Scholar can be int,
    str, or None (the API has been observed returning all three forms).
    The buggy code does `paper.get("citationCount") or 0` — if the value
    is a string like `"42"`, downstream math
    (`math.log10(max(citation_count, 1))`, `(citation_count - 10) // 50`)
    raises `TypeError: '>' not supported between 'str' and 'int'`.

    The fix must cast to int with a try/except (mirroring search_patents.py).
    """
    cases = [
        ("int form", 42),
        ("string form", "42"),
        ("None form", None),
        ("missing form", "missing"),
    ]

    for label, cc in cases:
        paper = _paper(citationCount=cc)
        # When "missing" — remove the key entirely
        if cc == "missing":
            paper = {k: v for k, v in paper.items() if k != "citationCount"}

        respx.get(sp.SEMANTIC_SCHOLAR_API).mock(
            return_value=httpx.Response(200, json=_s2_response([paper]))
        )
        # Must NOT raise — must gracefully return results with a sane int count.
        results = await sp.search_papers(_query())
        assert len(results) == 1, (
            f"[{label}] search_papers must return 1 result, got {len(results)}"
        )
        assert isinstance(results[0].citation_count, int), (
            f"[{label}] citation_count must be a Python int, got "
            f"{type(results[0].citation_count).__name__}: {results[0].citation_count!r}"
        )
        assert results[0].citation_count >= 0, (
            f"[{label}] citation_count must be ≥ 0, got {results[0].citation_count}"
        )


# ===========================================================================
# BUG #14: replication_count fabricated from citation_count
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_no_fabricated_replication_count():
    """SWARM-5 BUG #14: `replication_count = max(0, (citation_count - 10) // 50)`
    is a FABRICATED heuristic, NOT a real replication count. A paper cited
    60 times has `replication_count=1` — but that doesn't mean it was
    independently replicated even once. This lies about the epistemic
    strength of the result and inflates the `bridge_ranker` evidence score
    (which uses `replication_count >= 5` and `>= 2` as boost thresholds).

    The fix must either source `replication_count` from the API (if
    Semantic Scholar ever exposes it) OR set it to 0/None (NOT fabricate).
    """
    # A paper with 60 citations → current buggy code gives replication_count=1
    paper = _paper(citationCount=60)
    respx.get(sp.SEMANTIC_SCHOLAR_API).mock(
        return_value=httpx.Response(200, json=_s2_response([paper]))
    )

    results = await sp.search_papers(_query())
    assert len(results) == 1
    rc = results[0].replication_count

    # The current buggy code fabricates `(60 - 10) // 50 = 1`. The fix must
    # NOT fabricate — `replication_count` must be 0 (or sourced from the API).
    assert rc == 0, (
        f"replication_count must NOT be fabricated from citation_count "
        f"(BUG #14). Expected 0, got {rc}. Semantic Scholar does not expose "
        f"replication_count — fabricating it from `citation_count` lies about "
        f"whether the paper was independently replicated."
    )

    # Fake-fix detection: a "fake fix" that sets replication_count = citation_count
    # (or any non-zero function of citation_count) would still be fabrication.
    # Verify the source code does NOT compute replication_count from citation_count.
    src = inspect.getsource(sp)
    # The buggy line: `replication_count = max(0, (citation_count - 10) // 50)`
    # Any expression that derives replication_count from citation_count is a fabrication.
    assert "replication_count = max(0, (citation_count" not in src, (
        "search_papers must NOT derive `replication_count` from `citation_count`. "
        "This is the fabrication heuristic that BUG #14 flags."
    )
    assert "(citation_count - " not in src or "replication_count" not in src.split("(citation_count - ")[0].split("\n")[-1], (
        "search_papers must NOT use `citation_count - N` to compute `replication_count`."
    )


# ===========================================================================
# BUG #11: no retry / exponential backoff
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_handles_arxiv_rate_limit():
    """SWARM-5 BUG #11: HTTP 429 (rate limit) from Semantic Scholar must
    trigger retry with exponential backoff, not fail on the first attempt.

    Semantic Scholar is well-known to rate-limit aggressively — without
    retry/backoff the paper-search branch returns empty under any load.
    """
    route = respx.get(sp.SEMANTIC_SCHOLAR_API).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json=_s2_response([_paper()])),
        ]
    )
    sleep_mock = AsyncMock()

    with patch("asyncio.sleep", new=sleep_mock):
        results = await sp.search_papers(_query())

    assert route.call_count >= 2, (
        f"search_papers must retry on HTTP 429; only {route.call_count} attempt(s)."
    )
    assert sleep_mock.await_count >= 1, (
        "search_papers must sleep (exponential backoff) between 429 retries."
    )
    assert isinstance(results, list)


@pytest.mark.asyncio
@respx.mock
async def test_handles_arxiv_timeout():
    """SWARM-5 BUG #11: a network timeout must NOT crash the function.
    It must return an empty list (graceful degradation)."""
    respx.get(sp.SEMANTIC_SCHOLAR_API).mock(
        side_effect=httpx.ConnectTimeout("timed out")
    )

    results = await sp.search_papers(_query())
    assert results == [], (
        "search_papers must return an empty list on timeout, not raise."
    )


@pytest.mark.asyncio
@respx.mock
async def test_handles_arxiv_malformed_xml():
    """When the API returns invalid JSON (or HTML error page from a proxy),
    search_papers must gracefully return an empty list, not raise
    JSONDecodeError up to the caller.

    The current buggy code only catches `httpx.RequestError` and
    `httpx.HTTPStatusError` — NOT `json.JSONDecodeError`.
    """
    respx.get(sp.SEMANTIC_SCHOLAR_API).mock(
        return_value=httpx.Response(200, content=b"<html>Bad Gateway</html>")
    )

    results = await sp.search_papers(_query())
    assert results == [], (
        "search_papers must return an empty list on malformed JSON, not raise."
    )


# ===========================================================================
# Pagination with continuation token
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_pagination_with_continuation_token():
    """When the Semantic Scholar API returns a `next` continuation token,
    the fixed code must fetch the next page. The current code does ONE
    call and ignores the continuation."""
    page1 = {
        "data": [_paper(title="Paper 1")],
        "total": 2,
        "next": "TOKEN-page-2",
        "offset": 0,
    }
    page2 = {
        "data": [_paper(title="Paper 2")],
        "total": 2,
        "offset": 1,
    }
    route = respx.get(sp.SEMANTIC_SCHOLAR_API).mock(
        side_effect=[
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]
    )

    results = await sp.search_papers(_query())

    # The fixed code must follow the `next` continuation token.
    assert route.call_count >= 2, (
        f"search_papers must follow the Semantic Scholar `next` continuation token; "
        f"only {route.call_count} call(s) made."
    )
    titles = [r.title for r in results]
    assert "Paper 1" in titles and "Paper 2" in titles, (
        f"results must contain papers from both pages; got: {titles}"
    )


# ===========================================================================
# Empty results
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_handles_empty_results():
    """When the API returns zero papers, the function must return an empty
    list, not crash."""
    respx.get(sp.SEMANTIC_SCHOLAR_API).mock(
        return_value=httpx.Response(200, json={"data": [], "total": 0})
    )

    results = await sp.search_papers(_query())
    assert results == [], (
        f"empty Semantic Scholar response must yield empty list, got {results}"
    )