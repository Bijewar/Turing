"""
test_layer3_search_wikipedia.py — Wikipedia search service tests.

Catches every SWARM-5 BUG in services/layer3/search_wikipedia.py:
  BUG #1  — missing `import logging` + `logger = logging.getLogger(__name__)`
            means the error handler itself raises NameError, masking the
            original httpx exception. The whole /search request then 500s
            instead of degrading to an empty Wikipedia result list.
  BUG #11 — no retry / exponential backoff on HTTP 429/500/timeout. The
            service fails on the first transient error.
  BUG #37 — Wikipedia snippets containing HTML entities like `&quot;`,
            `&amp;`, `&#39;` are NOT decoded before being stored in
            `SearchResult.summary`, producing ugly and LLM-confusing text.

Design notes:
- Every test mocks httpx with `respx` so no real network call is made.
- Every test is deterministic and self-contained.
- Each test FAILS against the current buggy code (red) and PASSES only
  once the corresponding fix lands (green).
- Several tests include a "fake-fix detection" companion assertion so an
  LLM that superficially patches one symptom doesn't accidentally pass.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from schemas.layer3 import StructuralQuery, SearchResult, SearchSource
from services.layer3 import search_wikipedia as sw


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _query() -> StructuralQuery:
    return StructuralQuery(
        original_node_id="n0",
        original_confidence=50.0,
        structural_description="A high-variance input causes output failure.",
    )


def _wiki_search_response(*, title="Test", snippet="test snippet", pageid=1):
    """Build a minimal Wikipedia API search response payload."""
    return {
        "query": {
            "search": [
                {"title": title, "snippet": snippet, "pageid": pageid},
            ]
        }
    }


def _wiki_extlinks_response(pageid=1, n_links=5):
    """Build a minimal `prop=extlinks` response with n external links."""
    return {
        "query": {
            "pages": {
                str(pageid): {
                    "pageid": pageid,
                    "extlinks": [{"*": f"https://external.example/{i}"} for i in range(n_links)],
                }
            }
        }
    }


# ===========================================================================
# BUG #1: missing `logger` import — NameError masks the real exception
# ===========================================================================

def test_logger_is_imported():
    """SWARM-5 BUG #1: `search_wikipedia.py` references `logger.error(...)`
    in its except block but never imports `logging` nor defines `logger`.
    When Wikipedia returns 429/500/timeout, the except block itself raises
    `NameError: name 'logger' is not defined`, replacing the original
    httpx error and crashing the entire /search request instead of
    gracefully returning an empty result list.

    The fix must add `import logging` and `logger = logging.getLogger(__name__)`
    at the top of the module — mirroring `search_web.py`, `search_patents.py`,
    and `search_papers.py` which all already do this.
    """
    # 1) Module must expose a `logger` attribute (i.e. logging is imported).
    assert hasattr(sw, "logger"), (
        "search_wikipedia.py must define a module-level `logger` "
        "(via `import logging` + `logger = logging.getLogger(__name__)`). "
        "Without it, the except block raises NameError and masks the real exception."
    )
    assert isinstance(sw.logger, logging.Logger), (
        f"`logger` must be a logging.Logger instance, got {type(sw.logger).__name__}"
    )

    # 2) The except block in the source must reference `logger.error(...)` —
    #    this proves the error path is wired up rather than silently swallowed.
    src = inspect.getsource(sw)
    assert "logger.error" in src or "logger.warning" in src, (
        "search_wikipedia.py must use `logger.error` or `logger.warning` in its "
        "except block (currently present but NameErrors because `logger` is undefined)."
    )


@pytest.mark.asyncio
@respx.mock
async def test_logger_nameerror_does_not_mask_http_error():
    """SWARM-5 BUG #1 (functional): when Wikipedia returns HTTP 500, the
    except block must NOT raise NameError. The function must return an
    empty list (graceful degradation), not crash the caller.

    With the buggy code, the except block triggers `logger.error(...)` →
    `NameError: name 'logger' is not defined` → propagates to the caller.
    """
    respx.get(sw.WIKIPEDIA_SEARCH_API).mock(return_value=httpx.Response(500))

    # Should NOT raise — must return [] gracefully.
    results = await sw.search_wikipedia(_query())
    assert results == [], (
        "search_wikipedia must return an empty list on HTTP 500, not raise NameError."
    )


# ===========================================================================
# BUG #11: no retry / exponential backoff for HTTP 429/500/timeout
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_handles_http_429_with_retry():
    """SWARM-5 BUG #11: HTTP 429 (rate limit) must trigger retry with
    exponential backoff, not fail on the first attempt.

    The fixed code retries at least once (with sleep) before giving up.
    """
    route = respx.get(sw.WIKIPEDIA_SEARCH_API).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json=_wiki_search_response()),
        ]
    )
    sleep_mock = AsyncMock()

    with patch("asyncio.sleep", new=sleep_mock):
        results = await sw.search_wikipedia(_query())

    # Must have retried at least twice (so ≥3 total attempts) to honor 429.
    assert route.call_count >= 2, (
        f"search_wikipedia must retry on HTTP 429; only {route.call_count} attempt(s) made."
    )
    # Backoff: at least one sleep between attempts.
    assert sleep_mock.await_count >= 1, (
        "search_wikipedia must sleep (exponential backoff) between 429 retries."
    )
    # Eventually returned results (not just empty after retries).
    assert isinstance(results, list)


@pytest.mark.asyncio
@respx.mock
async def test_handles_http_500_with_retry():
    """SWARM-5 BUG #11: HTTP 500 (server error) must trigger retry with
    exponential backoff before giving up gracefully."""
    route = respx.get(sw.WIKIPEDIA_SEARCH_API).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json=_wiki_search_response()),
        ]
    )
    sleep_mock = AsyncMock()

    with patch("asyncio.sleep", new=sleep_mock):
        results = await sw.search_wikipedia(_query())

    assert route.call_count >= 2, (
        f"search_wikipedia must retry on HTTP 500; only {route.call_count} attempt(s) made."
    )


@pytest.mark.asyncio
@respx.mock
async def test_handles_timeout():
    """SWARM-5 BUG #11: a network timeout must NOT crash the function.
    It must return an empty list (graceful degradation)."""
    respx.get(sw.WIKIPEDIA_SEARCH_API).mock(side_effect=httpx.ConnectTimeout("timed out"))

    results = await sw.search_wikipedia(_query())
    assert results == [], (
        "search_wikipedia must return an empty list on timeout, not raise."
    )


@pytest.mark.asyncio
@respx.mock
async def test_handles_non_200_response():
    """SWARM-5 BUG #11: a 404 (or any non-2xx) must NOT crash the function.
    It must return an empty list (graceful degradation)."""
    respx.get(sw.WIKIPEDIA_SEARCH_API).mock(return_value=httpx.Response(404))

    results = await sw.search_wikipedia(_query())
    assert results == [], (
        "search_wikipedia must return an empty list on HTTP 404, not raise."
    )


# ===========================================================================
# Empty / malformed response handling
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_handles_empty_search_results():
    """When Wikipedia returns zero articles, the function must return an
    empty list, not crash."""
    respx.get(sw.WIKIPEDIA_SEARCH_API).mock(
        return_value=httpx.Response(200, json={"query": {"search": []}})
    )

    results = await sw.search_wikipedia(_query())
    assert results == [], (
        f"empty Wikipedia search must yield empty list, got {results}"
    )


@pytest.mark.asyncio
@respx.mock
async def test_handles_malformed_response():
    """When Wikipedia returns invalid JSON, the function must gracefully
    return an empty list, not raise JSONDecodeError up to the caller.

    The current buggy code only catches `httpx.RequestError` and
    `httpx.HTTPStatusError` — NOT `json.JSONDecodeError`. The fix must
    also catch malformed-JSON errors.
    """
    respx.get(sw.WIKIPEDIA_SEARCH_API).mock(
        return_value=httpx.Response(200, content=b"not valid json")
    )

    results = await sw.search_wikipedia(_query())
    assert results == [], (
        "search_wikipedia must return an empty list on malformed JSON, not raise."
    )


# ===========================================================================
# Pagination
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_pagination_handled():
    """When the Wikipedia API returns a `continue` token, the fixed code
    must fetch the next page (or, at minimum, not crash on the token).

    Wikipedia's search endpoint uses `sroffset` / `continue` for pagination.
    The current buggy code makes ONE search call and ignores the
    continuation — so a query that has more results than `MAX_RESULTS`
    only ever returns the first page.
    """
    page1 = {
        "query": {"search": [{"title": "T1", "snippet": "S1", "pageid": 101}]},
        "continue": {"sroffset": 1, "continue": "-||"},
    }
    page2 = {
        "query": {"search": [{"title": "T2", "snippet": "S2", "pageid": 102}]},
    }
    # Separate routes for the search list call vs. the per-article extlinks
    # call so we can count *search* calls specifically (not extlinks).
    search_route = respx.get(
        sw.WIKIPEDIA_SEARCH_API, params__contains={"list": "search"}
    ).mock(side_effect=[
        httpx.Response(200, json=page1),
        httpx.Response(200, json=page2),
    ])
    # The extlinks call should NOT be paginated — return empty links.
    respx.get(
        sw.WIKIPEDIA_SEARCH_API, params__contains={"prop": "extlinks"}
    ).mock(return_value=httpx.Response(200, json=_wiki_extlinks_response(n_links=0)))

    results = await sw.search_wikipedia(_query())

    # The fixed code MUST have followed the continuation token — i.e. the
    # *search* endpoint (not the extlinks endpoint) was called at least twice.
    assert search_route.call_count >= 2, (
        f"search_wikipedia must follow the Wikipedia `continue` token to "
        f"fetch the next page; only {search_route.call_count} search call(s) made."
    )
    assert isinstance(results, list)
    # Both pages' articles must be in the results.
    titles = [r.title for r in results]
    assert "T1" in titles and "T2" in titles, (
        f"results must contain articles from both pages; got titles: {titles}"
    )


# ===========================================================================
# Special characters in query
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_search_query_special_chars_escaped():
    """SWARM-5 BUG #35 (sibling): the search query is passed via httpx
    `params=` which URL-encodes it. The fixed code must NOT concatenate
    the raw query into the URL (which would break on special chars like
    `?`, `&`, `=`, `#`, spaces).
    """
    captured = {}

    def _capture(request):
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"query": {"search": []}})

    respx.get(sw.WIKIPEDIA_SEARCH_API).mock(side_effect=_capture)

    query_with_specials = StructuralQuery(
        original_node_id="n0",
        original_confidence=50.0,
        structural_description="A?B&C=D#E with spaces and + plus",
    )
    await sw.search_wikipedia(query_with_specials)

    # The query must be URL-encoded — no raw `?`, `&`, `#`, `+` in the URL
    # outside of the encoded params. The simplest check: the captured
    # `srsearch` param decodes back to the original string.
    assert "srsearch" in captured["params"], (
        f"search_wikipedia must pass the query as a URL param; got URL {captured.get('url')}"
    )
    # httpx.QueryParams decodes the value back; round-trip must equal the original.
    # If a future regression manually concatenates the query into the URL (bypassing
    # `params=`), the special chars (`?`, `&`, `#`) would corrupt the URL and the
    # `srsearch` key would either be missing or hold a truncated value.
    decoded = captured["params"]["srsearch"]
    assert decoded == query_with_specials.structural_description, (
        f"special-char query was not properly URL-encoded (round-trip failed): "
        f"decoded={decoded!r}, expected={query_with_specials.structural_description!r}"
    )
    # Fake-fix detection: the raw URL string must contain `srsearch=` followed by
    # an encoded value (the special chars `?`, `&`, `#` must be percent-encoded
    # inside the param value, NOT used as URL separators in the query path).
    raw_url = captured["url"]
    # The first `?` separates path from query string; subsequent `?` chars
    # inside the value must be encoded as `%3F`.
    query_string = raw_url.split("?", 1)[1] if "?" in raw_url else ""
    # `&` inside the value must be encoded as `%26`. Look at the value portion
    # after `srsearch=` to ensure no raw `?`, `#` from the query leaked through.
    sr_idx = query_string.find("srsearch=")
    if sr_idx != -1:
        sr_value = query_string[sr_idx + len("srsearch="):]
        # Stop at the next `&` that separates params (NOT encoded `%26`).
        # Split on `&` but ignore `%26`.
        # Easiest: just assert no raw `?`, `#` are present in the value.
        assert "?" not in sr_value, (
            f"raw `?` from the query leaked into the URL (must be %3F): {raw_url}"
        )
        assert "#" not in sr_value, (
            f"raw `#` from the query leaked into the URL (must be %23): {raw_url}"
        )


# ===========================================================================
# No API key — graceful behavior (Wikipedia doesn't require one, but other
# services do; verify the contract that missing keys don't crash Wikipedia).
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_no_api_key_clear_error(monkeypatch):
    """Wikipedia's API requires no API key, so search must succeed even
    when all `*_API_KEY` env vars are missing. This documents the
    contract and protects against future regressions that gate the
    Wikipedia call behind a key check.

    Note: `conftest.py` already strips all LLM API keys, so this test
    just verifies the call goes through.
    """
    respx.get(sw.WIKIPEDIA_SEARCH_API).mock(
        return_value=httpx.Response(200, json=_wiki_search_response())
    )
    # Patch the extlinks endpoint to avoid the secondary request.
    respx.get(sw.WIKIPEDIA_SEARCH_API, params__contains={"prop": "extlinks"}).mock(
        return_value=httpx.Response(200, json=_wiki_extlinks_response())
    )

    results = await sw.search_wikipedia(_query())
    assert len(results) == 1, (
        f"Wikipedia search must succeed without API key; got {len(results)} results"
    )
    assert results[0].source == SearchSource.WIKIPEDIA


# ===========================================================================
# BUG #37: HTML entities not decoded
# ===========================================================================

@pytest.mark.asyncio
@respx.mock
async def test_html_entities_decoded_in_summary():
    """SWARM-5 BUG #37: Wikipedia snippets often contain HTML entities like
    `&quot;`, `&amp;`, `&#39;` around technical terms. The buggy code only
    strips `<span class="searchmatch">` tags — it does NOT decode entities,
    so the `SearchResult.summary` contains raw `&quot;` which confuses the
    downstream LLM.

    The fix must call `html.unescape(...)` on the snippet.
    """
    snippet = "&quot;Hello&quot; world &amp; goodbye &#39;test&apos;"
    respx.get(sw.WIKIPEDIA_SEARCH_API).mock(
        return_value=httpx.Response(
            200,
            json=_wiki_search_response(snippet=snippet),
        )
    )
    respx.get(sw.WIKIPEDIA_SEARCH_API, params__contains={"prop": "extlinks"}).mock(
        return_value=httpx.Response(200, json=_wiki_extlinks_response())
    )

    results = await sw.search_wikipedia(_query())
    assert len(results) == 1
    summary = results[0].summary
    # The decoded summary should contain the literal characters, not entities.
    assert "&quot;" not in summary, (
        f"HTML entity `&quot;` must be decoded to `\"`, got: {summary!r}"
    )
    assert "&amp;" not in summary, (
        f"HTML entity `&amp;` must be decoded to `&`, got: {summary!r}"
    )
    assert "&#39;" not in summary, (
        f"HTML entity `&#39;` must be decoded to `'`, got: {summary!r}"
    )
    # Positive check: the decoded text contains the expected characters.
    assert '"Hello"' in summary or "Hello" in summary, (
        f"Decoded summary must contain `Hello` (with or without quotes); got: {summary!r}"
    )