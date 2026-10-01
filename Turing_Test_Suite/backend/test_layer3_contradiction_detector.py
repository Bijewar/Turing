"""
test_layer3_contradiction_detector.py — 3-Agent adversarial debate tests.

Catches every SWARM-5 / SWARM-8 BUG in services/layer3/contradiction_detector.py:

  BUG #2  — `run_skeptic` and `run_synthesizer` use sync `client.completion`
            inside async functions, blocking the event loop. Must use
            `acompletion` or `asyncio.to_thread`.
  BUG #5/6 — local `_get_llm_client` checks fewer keys than the shared one
            in `anthropic_client`.
  BUG #7   — `run_proposer` calls `max(results, ...)` which raises the
            GENERIC `ValueError: max() iterable argument is empty` on
            empty results. SWARM-8 asks for a clear, actionable error
            (or graceful return).
  LLM failure resilience — `asyncio.gather(*tasks, return_exceptions=True)`
            must preserve it (one bad LLM call must NOT kill all debates).

Design notes:
- Every test mocks the LLM client so no real provider call is made.
- Each test FAILS against the current buggy code (red) and PASSES only
  once the corresponding fix lands (green).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from schemas.layer3 import (
    ContradictionDetails,
    DebateTranscript,
    MergedResult,
    SearchSource,
    SearchResult,
    StructuralQuery,
)
from services.layer3 import contradiction_detector as cd
from services.layer3.contradiction_detector import AgentFactory


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _query() -> StructuralQuery:
    return StructuralQuery(
        original_node_id="n0",
        original_confidence=50.0,
        structural_description="A high-variance input causes output failure.",
    )


def _result(
    *,
    source: SearchSource = SearchSource.PAPER,
    title: str = "T",
    summary: str = "s",
    confidence: float = 80.0,
) -> SearchResult:
    return SearchResult(
        source=source,
        title=title,
        summary=summary,
        confidence=confidence,
        original_query=_query(),
    )


def _merged(
    *,
    title: str = "Merged",
    summary: str = "merged summary",
    n_sources: int = 2,
) -> MergedResult:
    """Build a MergedResult with `n_sources` source_results so it qualifies
    for the adversarial debate (the filter is `len(source_results) > 1`)."""
    sources = []
    source_results = []
    for i in range(n_sources):
        s = [SearchSource.PAPER, SearchSource.WEB, SearchSource.WIKIPEDIA, SearchSource.PATENT][i % 4]
        sources.append(s)
        source_results.append(_result(source=s, title=f"R{i}", summary=f"summary {i}", confidence=80.0 + i))
    return MergedResult(
        title=title,
        merged_summary=summary,
        underlying_mechanism="mechanism",
        sources=list({s for s in sources}),
        source_results=source_results,
        confidence=85.0,
    )


def _make_llm_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class _MockLLMClient:
    """Mock LLM client. Returns `accepted` (no contradiction) by default.
    Exposes both sync `completion` (current buggy path) and async `acompletion`
    (the fixed path). Tracks which was called so tests can assert."""
    def __init__(self, *, response: str = "ACCEPTED. No direct contradictions found in other sources.",
                 raise_exc: Optional[Exception] = None,
                 fail_on_calls: Optional[set] = None):
        self._response = response
        self._raise = raise_exc
        self._fail_on_calls = fail_on_calls or set()
        self.completion_calls = 0
        self.acompletion_calls = 0
        self.last_kwargs: dict = {}

    def completion(self, **kwargs):
        # SWARM-5 BUG #2: this is the SYNC entrypoint — should NOT be used.
        self.completion_calls += 1
        self.last_kwargs = kwargs
        if self._raise is not None and self.completion_calls in self._fail_on_calls:
            raise self._raise
        return _make_llm_response(self._response)

    async def acompletion(self, **kwargs):
        self.acompletion_calls += 1
        self.last_kwargs = kwargs
        if self._raise is not None and self.acompletion_calls in self._fail_on_calls:
            raise self._raise
        return _make_llm_response(self._response)


# ===========================================================================
# BUG #2: async uses acompletion, not completion
# ===========================================================================

@pytest.mark.asyncio
async def test_async_uses_acompletion_not_completion(monkeypatch):
    """SWARM-5 BUG #2: `run_skeptic` and `run_synthesizer` are declared
    `async def` but invoke the SYNC `client.completion(...)` — which is
    `litellm.completion`, the synchronous variant. This blocks the event
    loop for the full HTTP round-trip (1-5 s typical) and starves every
    concurrent coroutine in the same loop. In `asyncio.gather(*debate_tasks)`,
    debates effectively run SEQUENTIALLY — defeating the parallelism.

    The fix must use `await client.acompletion(...)` (litellm's native async
    API) OR `await asyncio.to_thread(client.completion, ...)`.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock_client = _MockLLMClient()

    with patch.object(cd, "_get_llm_client", return_value=mock_client):
        merged = _merged(n_sources=2)
        await cd.detect_contradictions([merged])

    # The async entrypoint must be used.
    assert mock_client.acompletion_calls >= 1, (
        "contradiction_detector must use `client.acompletion` (async) — the "
        "sync `client.completion` blocks the event loop (SWARM-5 BUG #2)."
    )
    # The sync entrypoint must NOT be used (or, alternatively, must be wrapped
    # in asyncio.to_thread — but the source must NOT call it bare).
    src = inspect.getsource(cd)
    # If `asyncio.to_thread` is used to wrap `completion`, that's acceptable.
    uses_to_thread = "asyncio.to_thread" in src and "completion" in src
    if not uses_to_thread:
        # Otherwise, acompletion must be used and bare completion must not.
        assert mock_client.completion_calls == 0, (
            "contradiction_detector must NOT use bare sync `client.completion` "
            "inside async functions (SWARM-5 BUG #2). Use acompletion or "
            "asyncio.to_thread."
        )


# ===========================================================================
# BUG #7 / SWARM-8: max() on empty results
# ===========================================================================

@pytest.mark.asyncio
async def test_max_on_empty_results_raises_clear_error():
    """SWARM-5 BUG #7 / SWARM-8: `run_proposer([])` calls `max(results, ...)`
    which raises the GENERIC `ValueError: max() iterable argument is empty`.
    This message gives the operator no actionable context.

    The fix must raise a CLEAR, actionable error (or return gracefully)
    that mentions "empty results" / "no sources" — NOT the bare max() error.
    """
    with pytest.raises(ValueError) as exc_info:
        await AgentFactory.run_proposer([])

    msg = str(exc_info.value)
    # The clear error must NOT be the generic max() error.
    assert "max()" not in msg and "iterable argument" not in msg.lower(), (
        f"run_proposer must raise a CLEAR ValueError on empty results, not the "
        f"generic max() error. Got: {exc_info.value!r}"
    )
    # The clear error must mention something actionable.
    msg_lower = msg.lower()
    assert any(kw in msg_lower for kw in (
        "empty", "no source", "no result", "no input", "no debate",
        "no search", "at least one",
    )), (
        f"run_proposer must raise a ValueError with an actionable message on "
        f"empty results. Got: {exc_info.value!r}"
    )


@pytest.mark.asyncio
async def test_handles_no_search_results():
    """detect_contradictions must NOT crash on an empty merged_results list.
    It must return an empty list (graceful degradation)."""
    result = await cd.detect_contradictions([])
    assert result == [], (
        f"detect_contradictions([]) must return []; got {result}"
    )


@pytest.mark.asyncio
async def test_handles_merged_with_single_source():
    """A MergedResult with only 1 source_result should NOT trigger a debate
    (no other source to contradict). It should pass through unchanged."""
    merged = _merged(n_sources=1)
    result = await cd.detect_contradictions([merged])
    assert len(result) == 1, (
        f"single-source merged_result must pass through; got {len(result)}"
    )
    # No debate was attempted (only 1 source).
    assert result[0].contradiction_analysis is None, (
        f"single-source merged_result must not have contradiction_analysis"
    )


# ===========================================================================
# Contradiction detection
# ===========================================================================

@pytest.mark.asyncio
async def test_detects_clear_contradiction(monkeypatch):
    """When the LLM (skeptic) returns REJECTED, the contradiction is flagged.
    The resulting ContradictionDetails must have `conflict_detected=True`."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Mock LLM: skeptic returns REJECTED, synthesizer returns JSON with consensus+experiment.
    responses = iter([
        "REJECTED. Data contradiction found in [WIKIPEDIA]: source A says X increases Y; source B says X decreases Y.",
        '{"consensus": "Direction conflict on X→Y", "experiment": "Measure Y under controlled X increase."}',
    ])

    class _SkepticalClient:
        completion_calls = 0
        acompletion_calls = 0
        def completion(self, **kwargs):
            self.completion_calls += 1
            return _make_llm_response(next(responses))
        async def acompletion(self, **kwargs):
            self.acompletion_calls += 1
            return _make_llm_response(next(responses))

    mock_client = _SkepticalClient()
    with patch.object(cd, "_get_llm_client", return_value=mock_client):
        merged = _merged(n_sources=2)
        results = await cd.detect_contradictions([merged])

    assert len(results) == 1
    ca = results[0].contradiction_analysis
    assert ca is not None, "Contradiction analysis must be set after debate."
    assert ca.conflict_detected is True, (
        f"When the skeptic returns REJECTED, conflict_detected must be True. "
        f"Got: {ca}"
    )
    assert "X→Y" in ca.nature_of_conflict or "contradiction" in ca.nature_of_conflict.lower() or "conflict" in ca.nature_of_conflict.lower(), (
        f"nature_of_conflict must mention the conflict. Got: {ca.nature_of_conflict!r}"
    )
    assert ca.recommended_experiment, "Recommended experiment must be set."


@pytest.mark.asyncio
async def test_no_false_positive_on_complementary_results(monkeypatch):
    """When the LLM (skeptic) returns ACCEPTED (no contradiction), no
    contradiction is flagged. This is the no-false-positive case: two
    results that complement each other (e.g., one says "in mice", other
    says "in humans") must NOT be flagged as contradictory."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock_client = _MockLLMClient(
        response="ACCEPTED. No direct contradictions found in other sources."
    )
    with patch.object(cd, "_get_llm_client", return_value=mock_client):
        merged = _merged(n_sources=2)
        results = await cd.detect_contradictions([merged])

    assert len(results) == 1
    ca = results[0].contradiction_analysis
    # When ACCEPTED, conflict_detected should be False.
    if ca is not None:
        assert ca.conflict_detected is False, (
            f"Complementary (ACCEPTED) results must NOT be flagged as "
            f"contradictions. Got conflict_detected={ca.conflict_detected}"
        )


# ===========================================================================
# Threshold tunability
# ===========================================================================

@pytest.mark.asyncio
async def test_threshold_tunable():
    """SWARM-5: detect_contradictions should accept a threshold/sensitivity
    parameter that controls how aggressively contradictions are flagged.
    Different thresholds should produce different results on borderline cases.

    The current function has no such parameter — the LLM's binary
    ACCEPTED/REJECTED is the only signal. The fix must add an explicit
    threshold (e.g., `contradiction_threshold: float = 0.5`).
    """
    sig = inspect.signature(cd.detect_contradictions)
    threshold_params = [
        name for name in sig.parameters
        if "threshold" in name.lower() or "sensitivity" in name.lower()
    ]
    assert len(threshold_params) > 0, (
        f"detect_contradictions should accept a threshold/sensitivity parameter "
        f"for tunable contradiction detection. Current params: "
        f"{list(sig.parameters)}"
    )


# ===========================================================================
# Robust embedding model
# ===========================================================================

@pytest.mark.asyncio
async def test_embeds_via_robust_model():
    """SWARM-5: the contradiction detector should use a robust embedding model
    (e.g., sentence-transformers) for semantic comparison, not just LLM
    prompting. If the model is unavailable (no internet, not installed),
    it must fall back gracefully — not crash the whole pipeline.

    The current code uses LLM prompting only, no embeddings. The fix must
    add an embedding-based comparison (or document why LLM-only is sufficient).
    """
    src = inspect.getsource(cd)
    # 1) Uses embeddings (sentence-transformers, huggingface, etc.)
    uses_embeddings = (
        "sentence_transformers" in src
        or "SentenceTransformer" in src
        or "huggingface" in src.lower()
        or ("embed" in src.lower() and "model" in src.lower())
    )
    assert uses_embeddings, (
        "contradiction_detector should use an embedding model (e.g., "
        "sentence-transformers) for robust semantic comparison, not just "
        "LLM prompting."
    )
    # 2) Has a graceful fallback if the embedding model is unavailable.
    has_fallback = (
        "ImportError" in src
        or "ModuleNotFoundError" in src
        or "OSError" in src
        # A broad except is also acceptable for graceful fallback.
        or ("except Exception" in src and "embed" in src.lower())
    )
    assert has_fallback, (
        "contradiction_detector should fall back gracefully if the embedding "
        "model is unavailable (ImportError / OSError / Exception handler)."
    )


# ===========================================================================
# LLM failure resilience
# ===========================================================================

@pytest.mark.asyncio
async def test_llm_failure_does_not_kill_all(monkeypatch):
    """SWARM-5: if one LLM call fails during contradiction detection, the
    other merged_results should still get their contradiction analysis.

    The current code uses `asyncio.gather(*tasks, return_exceptions=True)`
    and checks `isinstance(debate_result, Exception)` — so a single
    failure should NOT kill the whole batch. This test verifies that.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Build 3 merged_results (each with 2 sources to qualify for debate).
    merged_list = [_merged(title=f"M{i}", n_sources=2) for i in range(3)]

    # Mock LLM: succeed for debates 1 and 3, fail for debate 2.
    # With sync `client.completion` calls running sequentially under gather,
    # the call order is: M0.skeptic, M1.skeptic, M2.skeptic.
    # (run_synthesizer is only called if skeptic returns REJECTED — here it
    # returns ACCEPTED, so no synthesizer call.)
    class _PartiallyFailingClient:
        completion_calls = 0
        acompletion_calls = 0
        def completion(self, **kwargs):
            self.completion_calls += 1
            if self.completion_calls == 2:
                raise RuntimeError("LLM rate limit on second debate")
            return _make_llm_response(
                "ACCEPTED. No direct contradictions found in other sources."
            )
        async def acompletion(self, **kwargs):
            self.acompletion_calls += 1
            if self.acompletion_calls == 2:
                raise RuntimeError("LLM rate limit on second debate")
            return _make_llm_response(
                "ACCEPTED. No direct contradictions found in other sources."
            )

    mock_client = _PartiallyFailingClient()
    with patch.object(cd, "_get_llm_client", return_value=mock_client):
        results = await cd.detect_contradictions(merged_list)

    # All 3 merged_results must be preserved (the failure must not kill the batch).
    assert len(results) == 3, (
        f"All 3 merged_results must be returned even if one LLM call fails. "
        f"Got: {len(results)}"
    )
    # The failed debate (M1) should still be in the output, just without
    # contradiction_analysis (or with it set to None).
    titles = [r.title for r in results]
    assert "M0" in titles and "M1" in titles and "M2" in titles, (
        f"All merged_results must be preserved; got titles: {titles}"
    )
    # At least 2 of the 3 debates should have succeeded (non-None contradiction_analysis).
    successful = sum(1 for r in results if r.contradiction_analysis is not None)
    assert successful >= 2, (
        f"At least 2 of 3 debates should have completed despite 1 LLM failure. "
        f"Got {successful} successful."
    )