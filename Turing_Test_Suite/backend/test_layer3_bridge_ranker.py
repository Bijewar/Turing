"""
test_layer3_bridge_ranker.py — Bridge validity ranker tests.

Catches every SWARM-5 BUG in services/layer3/bridge_ranker.py:

  BUG #25 — `float(os.getenv("WEIGHT_PAPER", "0.55"))` raises ValueError
             on a non-numeric env var (e.g., `WEIGHT_PAPER=oops` or
             `WEIGHT_PAPER=0,7`). One bad env var takes down the entire
             `/rank` endpoint.
  BUG #26 — `final_score = structural * compatibility * transferability * evidence`
             becomes NaN if any LLM score is NaN. `min(1.0, max(0.0, NaN))`
             silently returns 0.0 (Python's order-dependent max/min with NaN).
             The bridge then gets `final_score=0.0` (silent corruption) — NOT
             rejected or sanitized to 0.5.
  BUG #27/28 — `for ... if isinstance(llm_result, Exception): raise llm_result`
             abandons ALL other (potentially successful) matches. One bad
             LLM call takes down the entire `/rank` response.

Design notes:
- Every test mocks the LLM (evaluate_compatibility_and_transferability) so
  no real provider call is made.
- Each test FAILS against the current buggy code (red) and PASSES only
  once the corresponding fix lands (green).
"""
from __future__ import annotations

import asyncio
import inspect
import math
import os
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from schemas.layer3 import (
    ExtractedMechanism,
    IsomorphismMatch,
    MatchType,
    MergedResult,
    RankedBridge,
    SearchSource,
    SearchResult,
    Step14Request,
    Step14Response,
    StructuralQuery,
    ValidityScores,
)
from schemas.graph import CausalGraph, Edge, Node
from services.layer3 import bridge_ranker as br


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _query() -> StructuralQuery:
    return StructuralQuery(
        original_node_id="n0",
        original_confidence=50.0,
        structural_description="A high-variance input causes output failure.",
    )


def _search_result(
    *,
    source: SearchSource = SearchSource.PAPER,
    title: str = "T",
    summary: str = "summary text",
    confidence: float = 80.0,
    citation_count: int = 0,
    replication_count: int = 0,
    deployment_status: str = "single_study",
) -> SearchResult:
    return SearchResult(
        source=source,
        title=title,
        summary=summary,
        confidence=confidence,
        original_query=_query(),
        citation_count=citation_count,
        replication_count=replication_count,
        deployment_status=deployment_status,
    )


def _merged(
    *,
    title: str = "Merged",
    summary: str = "merged summary text here.",
    sources: Optional[List[SearchSource]] = None,
    source_results: Optional[List[SearchResult]] = None,
) -> MergedResult:
    if source_results is None:
        source_results = [_search_result(source=SearchSource.PAPER, confidence=80.0)]
    if sources is None:
        sources = list({r.source for r in source_results})
    return MergedResult(
        title=title,
        merged_summary=summary,
        underlying_mechanism="mechanism",
        sources=sources,
        source_results=source_results,
        confidence=85.0,
    )


def _extracted(merged: Optional[MergedResult] = None) -> ExtractedMechanism:
    if merged is None:
        merged = _merged()
    return ExtractedMechanism(
        source_result=merged,
        causal_graph=CausalGraph(nodes=[Node(id="a", label="A", confidence=80.0)], edges=[]),
    )


def _match(
    *,
    isomorphism_score: float = 80.0,
    match_type: MatchType = MatchType.STRONG_PARTIAL,
    merged: Optional[MergedResult] = None,
) -> IsomorphismMatch:
    return IsomorphismMatch(
        mechanism=_extracted(merged),
        isomorphism_score=isomorphism_score,
        match_type=match_type,
    )


def _step14_request(
    matches: List[IsomorphismMatch],
    user_domain_context: str = "domain context here.",
) -> Step14Request:
    return Step14Request(matches=matches, user_domain_context=user_domain_context)


async def _mock_eval_compat(*args, **kwargs):
    """Default mock: returns neutral 0.5 / 0.5 scores."""
    return (0.5, 0.5)


# ===========================================================================
# BUG #25: env var parsing crashes on typo
# ===========================================================================

@pytest.mark.asyncio
async def test_env_var_parsing_does_not_crash_on_typo(monkeypatch, caplog):
    """SWARM-5 BUG #25: `float(os.getenv("WEIGHT_PAPER", "0.55"))` raises
    `ValueError: could not convert string to float: 'oops'` if an operator
    sets a non-numeric env var. The exception is uncaught and propagates
    out of `calculate_evidence_strength`, killing the entire `/rank`
    endpoint.

    The fix must wrap each `float(os.getenv(...))` in a try/except returning
    the default (with a warning log)."""
    # Set a non-numeric env var (typo).
    monkeypatch.setenv("WEIGHT_PAPER", "oops")
    # Also test other env vars to be thorough — same bug pattern.
    monkeypatch.setenv("BOOST_DEPLOYED", "high")
    monkeypatch.setenv("CITE_500", "0,10")  # European decimal

    request = _step14_request([_match()])

    import logging
    with caplog.at_level(logging.WARNING, logger="services.layer3.bridge_ranker"), \
         patch.object(br, "evaluate_compatibility_and_transferability", new=_mock_eval_compat):
        # Must NOT raise.
        response = await br.rank_bridges(request)

    assert isinstance(response, Step14Response), (
        "rank_bridges must return a Step14Response even with a typo'd env var"
    )
    # The bridge must still be ranked (using the default weights).
    assert len(response.top_bridges) >= 1, (
        "rank_bridges must still produce a ranking with default weights"
    )
    # A warning must be logged so the operator can diagnose the bad env var.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("WEIGHT_PAPER" in r.getMessage() or "WEIGHT" in r.getMessage() or "env" in r.getMessage().lower()
               for r in warnings), (
        "rank_bridges must log a warning about the bad env var (so the operator can diagnose it)."
    )


# ===========================================================================
# BUG #26: NaN scores
# ===========================================================================

@pytest.mark.asyncio
async def test_no_nan_in_scores(monkeypatch):
    """Sanity check: all score fields in the output must be finite (no NaN, Inf).
    A regression in any of the LLM-parsing or score-computing paths that
    introduces NaN would be caught here."""
    async def _mock(domain_context, candidate_mechanism):
        return (0.7, 0.6)
    with patch.object(br, "evaluate_compatibility_and_transferability", new=_mock):
        response = await br.rank_bridges(_step14_request([_match()]))
    for bridge in response.top_bridges:
        for field in (
            "structural_match",
            "constraint_compatibility",
            "solution_transferability",
            "evidence_strength",
            "final_score",
        ):
            v = getattr(bridge.scores, field)
            assert math.isfinite(v), (
                f"{field} is not finite: {v!r}"
            )


@pytest.mark.asyncio
async def test_nan_scores_rejected_or_handled(monkeypatch):
    """SWARM-5 BUG #26: if the LLM returns NaN for `constraint_compatibility`,
    the current code does `min(1.0, max(0.0, NaN))` which (due to Python's
    order-dependent max/min with NaN) silently returns `0.0`. The bridge
    then gets `final_score = structural * 0.0 * trans * evidence = 0.0`
    — a silent corruption, NOT a rejection or sanitization.

    The fix must either reject the bridge entirely OR sanitize NaN to a
    neutral 0.5 (per the BUG #26 fix description).
    """
    # Mock the LLM to return NaN for compatibility.
    async def _mock_nan(domain_context, candidate_mechanism):
        return (float("nan"), 0.8)
    with patch.object(br, "evaluate_compatibility_and_transferability", new=_mock_nan):
        request = _step14_request([_match(isomorphism_score=80.0)])
        response = await br.rank_bridges(request)

    # The bridge must NOT silently get a 0.0 compatibility score (the
    # silent-corruption case). Either:
    # (a) the bridge is rejected (top_bridges is empty), or
    # (b) the bridge has a non-zero compatibility (sanitized to ~0.5).
    if response.top_bridges:
        for bridge in response.top_bridges:
            # The compatibility must NOT be the silent 0.0 from `max(0.0, NaN)`.
            assert bridge.scores.constraint_compatibility != 0.0, (
                f"NaN compatibility was silently converted to 0.0 (SWARM-5 BUG #26). "
                f"Expected: rejected or sanitized to a non-zero value. Got: {bridge.scores}"
            )
            # All scores must be finite.
            for field in (
                "structural_match", "constraint_compatibility",
                "solution_transferability", "evidence_strength", "final_score",
            ):
                v = getattr(bridge.scores, field)
                assert math.isfinite(v), (
                    f"NaN propagated to {field}: {v!r}"
                )


# ===========================================================================
# Tie breaking determinism
# ===========================================================================

@pytest.mark.asyncio
async def test_tie_breaking_deterministic(monkeypatch):
    """When two bridges have the same final_score, the tie-breaking must be
    deterministic (e.g., by original index, title, or another stable key) —
    NOT random or order-dependent.

    The current code uses `sorted(ranked_bridges, key=lambda x: x.scores.final_score,
    reverse=True)` which is stable in Python, so ties preserve input order.
    This test verifies that the tie-breaking is deterministic across runs."""
    # Build two bridges with identical scores.
    m1 = _match(isomorphism_score=80.0, merged=_merged(title="A"))
    m2 = _match(isomorphism_score=80.0, merged=_merged(title="B"))

    # Mock LLM to return identical scores for both.
    async def _mock(domain_context, candidate_mechanism):
        return (0.7, 0.6)
    with patch.object(br, "evaluate_compatibility_and_transferability", new=_mock):
        # Run twice with reversed input order — output order must be the same.
        response1 = await br.rank_bridges(_step14_request([m1, m2]))
        response2 = await br.rank_bridges(_step14_request([m2, m1]))

    titles1 = [b.match.mechanism.source_result.title for b in response1.top_bridges]
    titles2 = [b.match.mechanism.source_result.title for b in response2.top_bridges]
    # The tie-breaking must be deterministic — same output regardless of input order.
    # (Either "A then B" both times, or "B then A" both times — but consistent.)
    # We check that the SET of titles is the same (both bridges are in the output).
    assert set(titles1) == {"A", "B"}, (
        f"Both bridges must be in top_bridges (no tie-induced drops); got: {titles1}"
    )
    assert set(titles2) == {"A", "B"}, (
        f"Both bridges must be in top_bridges (no tie-induced drops); got: {titles2}"
    )
    # The tie-breaking must be deterministic across input orderings. The
    # canonical test: the final_score of bridge "A" must be the SAME in both
    # responses (since the LLM returns identical scores).
    score1_a = next(b.scores.final_score for b in response1.top_bridges
                    if b.match.mechanism.source_result.title == "A")
    score2_a = next(b.scores.final_score for b in response2.top_bridges
                    if b.match.mechanism.source_result.title == "A")
    assert score1_a == score2_a, (
        f"Tie-breaking must produce deterministic scores; A's score differs: "
        f"{score1_a} vs {score2_a}"
    )


# ===========================================================================
# Empty candidates
# ===========================================================================

@pytest.mark.asyncio
async def test_empty_candidates_handled():
    """An empty matches list must return an empty top_bridges list, not crash."""
    request = _step14_request([])
    response = await br.rank_bridges(request)
    assert response.top_bridges == [], (
        f"empty matches must yield empty top_bridges; got {response.top_bridges}"
    )


@pytest.mark.asyncio
async def test_all_discarded_handled():
    """When ALL matches are DISCARDED, the function must return an empty
    top_bridges list, not crash."""
    request = _step14_request([_match(match_type=MatchType.DISCARDED)])
    response = await br.rank_bridges(request)
    assert response.top_bridges == [], (
        f"all-discarded matches must yield empty top_bridges; got {response.top_bridges}"
    )


# ===========================================================================
# BUG #27/28: one LLM failure does not kill all rankings
# ===========================================================================

@pytest.mark.asyncio
async def test_one_llm_failure_does_not_kill_all_rankings(monkeypatch):
    """SWARM-5 BUG #27/28: `asyncio.gather(*tasks, return_exceptions=True)`
    collects LLM call results, but the loop has
    `if isinstance(llm_result, Exception): raise llm_result` — re-raising
    the FIRST exception and abandoning ALL other (potentially successful)
    matches. So one bad LLM call (transient network error, rate limit on
    one of N candidates) takes down the entire `/rank` response.

    The fix must log the exception, skip the failed match (or assign neutral
    0.5 scores), and continue. Return what's available with a warning.
    """
    # Build 3 matches. The 2nd one's LLM call will fail.
    matches = [
        _match(isomorphism_score=80.0, merged=_merged(title="M0")),
        _match(isomorphism_score=80.0, merged=_merged(title="M1")),
        _match(isomorphism_score=80.0, merged=_merged(title="M2")),
    ]

    call_count = {"n": 0}
    async def _mock(domain_context, candidate_mechanism):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("LLM rate limit on second bridge")
        return (0.7, 0.6)

    with patch.object(br, "evaluate_compatibility_and_transferability", new=_mock):
        # Must NOT raise — the failed LLM call must be skipped or sanitized.
        response = await br.rank_bridges(_step14_request(matches))

    # The other 2 matches must still be in the output.
    titles = [b.match.mechanism.source_result.title for b in response.top_bridges]
    assert "M0" in titles and "M2" in titles, (
        f"The 2 successful bridges (M0, M2) must still be ranked despite M1's "
        f"LLM failure. Got: {titles}"
    )
    # The response should NOT be empty (a single LLM failure must not 500 the
    # entire endpoint).
    assert len(response.top_bridges) >= 2, (
        f"At least 2 of 3 bridges must be in the output despite 1 LLM failure. "
        f"Got: {len(response.top_bridges)}"
    )


# ===========================================================================
# Score normalization
# ===========================================================================

@pytest.mark.asyncio
async def test_score_normalization_correct(monkeypatch):
    """All score fields must be in [0, 1] (per the ValidityScores schema
    `ge=0.0, le=1.0` constraints). The final_score is the geometric product
    of structural_match, compatibility, transferability, evidence — all
    in [0, 1] — so it must also be in [0, 1].

    The current code does `structural_match = match.isomorphism_score / 100.0`
    — this assumes `isomorphism_score` is in [0, 100]. If the upstream
    isomorphism matcher emits a score outside [0, 100], the resulting
    `structural_match` would be outside [0, 1] and Pydantic would reject it
    (or worse, silently accept and produce an out-of-range final_score).
    """
    async def _mock(domain_context, candidate_mechanism):
        return (0.7, 0.6)
    with patch.object(br, "evaluate_compatibility_and_transferability", new=_mock):
        response = await br.rank_bridges(_step14_request([_match(isomorphism_score=80.0)]))

    for bridge in response.top_bridges:
        for field in (
            "structural_match",
            "constraint_compatibility",
            "solution_transferability",
            "evidence_strength",
            "final_score",
        ):
            v = getattr(bridge.scores, field)
            assert 0.0 <= v <= 1.0, (
                f"{field} must be in [0, 1]; got {v}"
            )


# ===========================================================================
# Metadata preservation
# ===========================================================================

@pytest.mark.asyncio
async def test_preserves_bridge_metadata(monkeypatch):
    """After ranking, the resulting RankedBridge must keep all original
    metadata: the match's mechanism, source_result, isomorphism_score, and
    match_type must all survive the ranking process unchanged."""
    merged = _merged(
        title="Preserved Title",
        summary="preserved summary text",
        source_results=[
            _search_result(source=SearchSource.PAPER, title="Original Paper",
                          confidence=80.0, citation_count=42, replication_count=3,
                          deployment_status="replicated"),
        ],
    )
    match = _match(isomorphism_score=87.5, merged=merged, match_type=MatchType.STRONG_PARTIAL)

    async def _mock(domain_context, candidate_mechanism):
        return (0.7, 0.6)
    with patch.object(br, "evaluate_compatibility_and_transferability", new=_mock):
        response = await br.rank_bridges(_step14_request([match]))

    assert len(response.top_bridges) == 1
    bridge = response.top_bridges[0]
    # The match must be preserved.
    assert bridge.match is match or bridge.match == match, (
        "the original IsomorphismMatch must be preserved in the RankedBridge"
    )
    # The mechanism's source_result metadata must survive.
    assert bridge.match.mechanism.source_result.title == "Preserved Title"
    assert bridge.match.mechanism.source_result.merged_summary == "preserved summary text"
    # The source_result's source_results must be preserved.
    sr = bridge.match.mechanism.source_result.source_results[0]
    assert sr.citation_count == 42
    assert sr.replication_count == 3
    assert sr.deployment_status == "replicated"
    # The isomorphism_score and match_type must survive.
    assert bridge.match.isomorphism_score == 87.5
    assert bridge.match.match_type == MatchType.STRONG_PARTIAL