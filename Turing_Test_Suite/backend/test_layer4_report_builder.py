r"""
test_layer4_report_builder.py — Layer 4 Final Report Generator tests.

Catches every SWARM-6 BUG in services/layer4/report_builder.py:
  BUG #31 — sync `client.completion(...)` inside `async def build_report`
  BUG #32 — no retry / backoff on transient LLM failures
  BUG #33 — `isomorphism_score * 100` produces absurd `8750%`
  BUG #34 — `response_format={"type":"json_object"}` sent to non-OpenAI providers
  BUG #35 — greedy regex `\{.*\}` picks wrong JSON object
  BUG #36 — large context not chunked / truncated → token overflow
  BUG #37 — `user_query` is actually `bridges[0].merged_summary[:300]`
  BUG #23 — API keys leaked via LLM error messages / report text

Design notes:
- Every test mocks the LLM client so no real provider call is made.
- Every test is deterministic.
- Tests are written so they FAIL on the current buggy code (red) and PASS only
  after the bug is properly fixed (green). Several tests include a "fake-fix
  detection" companion assertion so a partial fix doesn't accidentally pass.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
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
    Step14Response,
    StructuralQuery,
    ValidityScores,
    DebateTranscript,
    ContradictionDetails,
)
from schemas.graph import CausalGraph, Edge, Node
from schemas.report import FinalReport


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _make_node(i: int = 0, label: Optional[str] = None) -> Node:
    return Node(id=f"n{i}", label=label or f"node-{i}", confidence=85.0)


def _make_edge(i: int = 0, src: str = "n0", tgt: str = "n1") -> Edge:
    return Edge(source=src, target=tgt, relation="ACTIVATES", confidence=80.0)


def _make_graph(n_nodes: int = 3) -> CausalGraph:
    nodes = [_make_node(i) for i in range(n_nodes)]
    edges = [_make_edge(i, f"n{i}", f"n{(i + 1) % n_nodes}") for i in range(n_nodes)]
    return CausalGraph(nodes=nodes, edges=edges)


def _make_structural_query() -> StructuralQuery:
    return StructuralQuery(
        original_node_id="n0",
        original_confidence=50.0,
        structural_description="A high-variance input causes output failure.",
    )


def _make_search_result() -> SearchResult:
    return SearchResult(
        source=SearchSource.WEB,
        title="Cross-domain paper",
        summary="A deployed production system.",
        url="https://example.com",
        confidence=80.0,
        original_query=_make_structural_query(),
        citation_count=10,
        replication_count=2,
        deployment_status="deployed",
    )


def _make_merged_result(
    title: str = "Cross-domain paper",
    merged_summary: Optional[str] = None,
    underlying_mechanism: str = "Bottleneck constriction mechanism.",
    contradiction: Optional[ContradictionDetails] = None,
) -> MergedResult:
    return MergedResult(
        title=title,
        merged_summary=merged_summary if merged_summary is not None else "Some merged summary text." * 3,
        underlying_mechanism=underlying_mechanism,
        sources=[SearchSource.WEB],
        source_results=[_make_search_result()],
        confidence=82.0,
        contradiction_analysis=contradiction,
    )


def _make_extracted_mechanism(merged_summary: Optional[str] = None) -> ExtractedMechanism:
    return ExtractedMechanism(
        source_result=_make_merged_result(merged_summary=merged_summary),
        causal_graph=_make_graph(3),
    )


def _make_isomorphism_match(
    isomorphism_score: float = 87.5,
    merged_summary: Optional[str] = None,
    match_type: MatchType = MatchType.STRONG_PARTIAL,
) -> IsomorphismMatch:
    return IsomorphismMatch(
        mechanism=_make_extracted_mechanism(merged_summary=merged_summary),
        isomorphism_score=isomorphism_score,
        match_type=match_type,
    )


def _make_validity_scores(final_score: float = 0.85) -> ValidityScores:
    return ValidityScores(
        structural_match=0.9,
        constraint_compatibility=0.85,
        solution_transferability=0.8,
        evidence_strength=0.9,
        final_score=final_score,
    )


def _make_bridge(
    isomorphism_score: float = 87.5,
    final_score: float = 0.85,
    merged_summary: Optional[str] = None,
    match_type: MatchType = MatchType.STRONG_PARTIAL,
) -> RankedBridge:
    return RankedBridge(
        match=_make_isomorphism_match(
            isomorphism_score=isomorphism_score,
            merged_summary=merged_summary,
            match_type=match_type,
        ),
        scores=_make_validity_scores(final_score=final_score),
    )


def _make_step14(bridges: List[RankedBridge]) -> Step14Response:
    return Step14Response(top_bridges=bridges)


def _make_llm_response(content: str):
    """Build a fake LiteLLM response object with the given content string."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Robust LLM mock — captures on BOTH `completion` (sync) and `acompletion`
# (async). Tests can inspect `.completion_calls`, `.acompletion_calls`,
# `.completion_kwargs`, and `.acompletion_kwargs` to determine which path
# the report builder took.
# ---------------------------------------------------------------------------
_DEFAULT_LLM_JSON = (
    '{"the_mechanism":"m","the_experiment":"e","who_solved_this":"w",'
    '"warnings_and_conflicts":[],"next_3_actions":["a","b","c"]}'
)


class _LLMMock:
    """Records every call (sync or async) and returns a configurable response.

    Use `raise_factory=ExceptionClass` to raise on every call. For per-attempt
    behaviour, pass a `side_effect` callable that returns either an Exception
    to raise or a content string to return.
    """
    def __init__(
        self,
        response_content: str = _DEFAULT_LLM_JSON,
        side_effect=None,
    ):
        self._content = response_content
        self._side_effect = side_effect
        self.completion_calls = 0
        self.acompletion_calls = 0
        self.completion_kwargs_list: List[dict] = []
        self.acompletion_kwargs_list: List[dict] = []

    def _resolve(self, kwargs):
        if self._side_effect is not None:
            res = self._side_effect(kwargs)
            if isinstance(res, Exception):
                raise res
            if isinstance(res, str):
                return _make_llm_response(res)
        return _make_llm_response(self._content)

    @property
    def completion_kwargs(self) -> dict:
        return self.completion_kwargs_list[-1] if self.completion_kwargs_list else {}

    @property
    def acompletion_kwargs(self) -> dict:
        return self.acompletion_kwargs_list[-1] if self.acompletion_kwargs_list else {}

    @property
    def last_kwargs(self) -> dict:
        """Whichever method was called last."""
        if self.acompletion_kwargs_list:
            return self.acompletion_kwargs_list[-1]
        if self.completion_kwargs_list:
            return self.completion_kwargs_list[-1]
        return {}

    @property
    def any_kwargs(self) -> dict:
        """The first non-empty kwargs dict from either method (prefer async)."""
        if self.acompletion_kwargs_list:
            return self.acompletion_kwargs_list[-1]
        if self.completion_kwargs_list:
            return self.completion_kwargs_list[-1]
        return {}

    @property
    def prompt(self) -> str:
        kw = self.any_kwargs
        msgs = kw.get("messages", [])
        if msgs:
            return msgs[0].get("content", "")
        return ""

    @property
    def total_calls(self) -> int:
        return self.completion_calls + self.acompletion_calls

    def completion(self, **kwargs):
        self.completion_calls += 1
        self.completion_kwargs_list.append(kwargs)
        return self._resolve(kwargs)

    async def acompletion(self, **kwargs):
        self.acompletion_calls += 1
        self.acompletion_kwargs_list.append(kwargs)
        return self._resolve(kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_build_report_uses_acompletion_not_completion():
    """SWARM-6 BUG #31: `build_report` is async and MUST `await client.acompletion(...)`,
    NOT call the sync `client.completion(...)` that blocks the event loop."""
    from services.layer4 import report_builder as rb

    mock_client = _LLMMock()
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        report = await rb.build_report(
            run_id="run-1",
            step14_response=_make_step14([_make_bridge()]),
        )

    # The async entrypoint must be used.
    assert mock_client.acompletion_calls >= 1, (
        "build_report must call client.acompletion(...) — the async entrypoint."
    )
    # The sync entrypoint must NOT be used.
    assert mock_client.completion_calls == 0, (
        "build_report must NOT call client.completion(...) — that blocks the event loop."
    )


@pytest.mark.asyncio
async def test_build_report_retries_on_transient_failure():
    """SWARM-6 BUG #32: A single transient LLM failure (e.g. HTTP 429) must NOT
    immediately trigger fallback. The fixed code retries at least once before
    falling back, so the LLM is called more than once."""
    from services.layer4 import report_builder as rb

    state = {"n": 0}

    def side_effect(kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return RuntimeError("rate_limit_429")  # raise on first call
        return _DEFAULT_LLM_JSON  # succeed on second

    mock_client = _LLMMock(side_effect=side_effect)
    with patch.object(rb, "_get_llm_client", return_value=mock_client), \
         patch("asyncio.sleep", new=AsyncMock()) as sleep_mock:
        report = await rb.build_report(
            run_id="run-retry",
            step14_response=_make_step14([_make_bridge()]),
        )

    # The fix must retry — at least two LLM calls happened.
    assert mock_client.total_calls >= 2, (
        f"build_report must retry on transient LLM failure; only {mock_client.total_calls} call(s) happened."
    )
    # And the second call's response was used, not the fallback.
    assert report.the_mechanism == "m", (
        f"Expected LLM response to be used after retry; got fallback: {report.the_mechanism[:100]!r}"
    )


@pytest.mark.asyncio
async def test_build_report_exponential_backoff():
    """SWARM-6 BUG #32 (backoff): When the LLM raises 429 three times in a row,
    the fixed code waits with increasing delays before each retry. The buggy
    code falls back immediately and never sleeps between retries."""
    from services.layer4 import report_builder as rb

    def side_effect(kwargs):
        return RuntimeError("rate_limit_429")  # always raise

    mock_client = _LLMMock(side_effect=side_effect)
    sleep_mock = AsyncMock()

    with patch.object(rb, "_get_llm_client", return_value=mock_client), \
         patch("asyncio.sleep", new=sleep_mock):
        report = await rb.build_report(
            run_id="run-backoff",
            step14_response=_make_step14([_make_bridge()]),
        )

    # Multiple attempts must have happened (not just one).
    assert mock_client.total_calls >= 3, (
        f"build_report must retry at least 3 times; only {mock_client.total_calls} call(s) happened."
    )
    # asyncio.sleep must have been awaited at least twice between retries.
    assert sleep_mock.await_count >= 2, (
        f"build_report must sleep between retries; only {sleep_mock.await_count} sleep(s) happened."
    )
    # Exponential backoff: delays must be (weakly) increasing across the first
    # two sleeps. A constant delay is NOT exponential backoff — it must grow.
    sleep_args = [
        call.args[0] if call.args else call.kwargs.get("delay", 0)
        for call in sleep_mock.await_args_list
    ]
    if len(sleep_args) >= 2:
        # Allow small jitter but require the trend to be increasing.
        assert sleep_args[1] >= sleep_args[0] * 1.5, (
            f"Backoff delays must increase exponentially; got {sleep_args}"
        )


@pytest.mark.asyncio
async def test_build_report_falls_back_only_after_max_retries():
    """SWARM-6 BUG #32 (fallback timing): Fallback must fire after `max_retries`
    attempts (e.g. 3), NOT after the first failure."""
    from services.layer4 import report_builder as rb

    def side_effect(kwargs):
        return RuntimeError("rate_limit_429")  # always raise

    mock_client = _LLMMock(side_effect=side_effect)
    with patch.object(rb, "_get_llm_client", return_value=mock_client), \
         patch("asyncio.sleep", new=AsyncMock()):
        report = await rb.build_report(
            run_id="run-max-retry",
            step14_response=_make_step14([_make_bridge()]),
        )

    # Fallback fired — we got a FinalReport back, not an exception.
    assert isinstance(report, FinalReport)
    # And it took at least 3 LLM attempts before falling back.
    assert mock_client.total_calls >= 3, (
        f"Fallback must fire only after ≥3 retries, not 1; "
        f"got {mock_client.total_calls} attempt(s)."
    )


@pytest.mark.asyncio
async def test_build_report_isomorphism_score_not_multiplied_by_100():
    """SWARM-6 BUG #33: `isomorphism_score` is documented as 0-100 in
    schemas/layer3.py. The buggy code computes `int(score * 100)` producing
    absurd values like 8750%. The fixed code displays the score as-is.

    The buggy code constructs `bridge_detail_lines` (which feed the LLM
    prompt) with `score_pct = int(bridge.match.isomorphism_score * 100)`.
    With `isomorphism_score=87.5`, this produces `8750`, which appears in
    the prompt as "8750% structural isomorphism" and (in the fallback path)
    in the report's `who_solved_this` as "8750% structural match".
    """
    from services.layer4 import report_builder as rb

    bridge = _make_bridge(isomorphism_score=87.5, final_score=0.85)
    mock_client = _LLMMock()
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        report = await rb.build_report(
            run_id="run-score",
            step14_response=_make_step14([bridge]),
        )

    # 1. The LLM prompt must NOT contain "8750%".
    prompt = mock_client.prompt
    assert "8750%" not in prompt, (
        f"isomorphism_score was multiplied by 100 in the prompt: 87.5 → 8750%. "
        f"Prompt excerpt: ...{prompt[prompt.find('Match:') - 10:prompt.find('Match:') + 50] if 'Match:' in prompt else prompt[:300]!r}..."
    )
    # 2. The user-facing report text must NOT contain "8750%".
    all_text = " ".join(
        [
            report.the_mechanism,
            report.the_experiment,
            report.who_solved_this,
            report.confidence_disclaimer,
            " ".join(report.next_3_actions or []),
            " ".join(report.warnings_and_conflicts or []),
        ]
    )
    assert "8750%" not in all_text, (
        f"isomorphism_score was multiplied by 100; 8750% leaked into report: {all_text[:200]}"
    )
    assert "8750" not in all_text, (
        f"Raw 8750 found in report text: {all_text[:200]}"
    )


@pytest.mark.asyncio
async def test_build_report_isomorphism_score_within_valid_range():
    """SWARM-6 BUG #33: Any isomorphism score displayed in the prompt or report
    must be a percentage in [0, 100], not [0, 8750]."""
    from services.layer4 import report_builder as rb

    bridge = _make_bridge(isomorphism_score=87.5, final_score=0.85)
    mock_client = _LLMMock()
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        report = await rb.build_report(
            run_id="run-range",
            step14_response=_make_step14([bridge]),
        )

    # Find all "N% structural" / "N% match" patterns in the prompt.
    prompt = mock_client.prompt
    pct_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:structural|match|isomorphism)", prompt
    )
    assert pct_matches, (
        "Expected at least one 'N% structural/match' pattern in the prompt, "
        f"got none. Prompt excerpt: {prompt[:400]!r}"
    )
    for pct_str in pct_matches:
        pct = float(pct_str)
        assert 0 <= pct <= 100, (
            f"Displayed match percentage {pct} is outside [0, 100]. "
            f"Prompt excerpt: {prompt[:400]!r}"
        )


@pytest.mark.asyncio
async def test_build_report_response_format_only_for_openai():
    """SWARM-6 BUG #34: `response_format={"type": "json_object"}` is the
    OpenAI JSON-mode flag. Anthropic / NVIDIA NIM / Groq do not support it.
    The fixed code must only set it when the configured model is OpenAI-family
    (gpt-*, openai/*, azure/*)."""
    from services.layer4 import report_builder as rb

    with patch.dict(os.environ, {"DEFAULT_LLM_MODEL": "claude-3-5-sonnet-20240620"}):
        mock_client = _LLMMock()
        with patch.object(rb, "_get_llm_client", return_value=mock_client):
            await rb.build_report(
                run_id="run-anthropic",
                step14_response=_make_step14([_make_bridge()]),
            )

        # The Claude/Anthropic call must NOT carry response_format.
        used_kwargs = mock_client.any_kwargs
        assert "response_format" not in used_kwargs, (
            f"response_format was passed to Anthropic model: {used_kwargs.get('response_format')!r}"
        )


@pytest.mark.asyncio
async def test_build_report_response_format_set_for_openai_model():
    """Companion test for SWARM-6 BUG #34 — for an OpenAI-family model,
    response_format={"type": "json_object"} SHOULD still be passed.

    This catches fake fixes that simply delete `response_format` for every
    model."""
    from services.layer4 import report_builder as rb

    with patch.dict(os.environ, {"DEFAULT_LLM_MODEL": "gpt-4o"}):
        mock_client = _LLMMock()
        with patch.object(rb, "_get_llm_client", return_value=mock_client):
            await rb.build_report(
                run_id="run-openai",
                step14_response=_make_step14([_make_bridge()]),
            )

        used_kwargs = mock_client.any_kwargs
        assert used_kwargs.get("response_format") == {"type": "json_object"}, (
            f"OpenAI-family model must receive response_format=json_object; got {used_kwargs.get('response_format')!r}"
        )


@pytest.mark.asyncio
async def test_build_report_json_extraction_not_greedy_regex():
    """SWARM-6 BUG #35: The buggy parser uses `re.search(r'\\{.*\\}', raw, re.DOTALL)`
    which is GREEDY — it captures from the first `{` to the LAST `}`, eating
    any commentary in between. The fixed parser must extract ONLY the FIRST
    valid JSON object."""
    from services.layer4 import report_builder as rb

    # LLM returns valid JSON, then commentary, then a second JSON object.
    raw = (
        '{"the_mechanism": "real mechanism", '
        '"the_experiment": "real experiment", '
        '"who_solved_this": "real bridges", '
        '"warnings_and_conflicts": [], '
        '"next_3_actions": ["a","b","c"]}'
        " Here is some commentary. Hope this helps! "
        '{"noise": "should be ignored"}'
    )
    mock_client = _LLMMock(response_content=raw)
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        report = await rb.build_report(
            run_id="run-parse",
            step14_response=_make_step14([_make_bridge()]),
        )

    # The parser must extract the FIRST JSON object's keys, not blend them
    # with the commentary or the second JSON object.
    assert "real mechanism" in report.the_mechanism, (
        f"First JSON object was not extracted; got: {report.the_mechanism[:200]}"
    )
    assert "commentary" not in report.the_mechanism.lower(), (
        f"Commentary leaked into the_mechanism: {report.the_mechanism[:200]}"
    )
    assert "should be ignored" not in report.the_mechanism, (
        f"Second JSON object leaked into the_mechanism: {report.the_mechanism[:200]}"
    )


@pytest.mark.asyncio
async def test_build_report_handles_no_json_in_llm_response():
    """SWARM-6 BUG #35 (graceful fallback): If the LLM returns prose only,
    `build_report` must NOT raise — it must return either a fallback report
    or a report with the raw text in `the_mechanism`."""
    from services.layer4 import report_builder as rb

    raw = "Sorry, I can't generate a JSON report right now. Please try again later."
    mock_client = _LLMMock(response_content=raw)
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        # Must not raise.
        report = await rb.build_report(
            run_id="run-no-json",
            step14_response=_make_step14([_make_bridge()]),
        )

    assert isinstance(report, FinalReport)


@pytest.mark.asyncio
async def test_build_report_truncates_large_context():
    """SWARM-6 BUG #36: With three large bridges (each ~10KB summary), the
    prompt sent to the LLM must be truncated to fit within the model's token
    limit. The buggy code passes the full `merged_summary` verbatim, so the
    prompt balloons past 30KB and the LLM call fails / falls back."""
    from services.layer4 import report_builder as rb

    # Three bridges, each with a 10KB merged_summary.
    bridges = [_make_bridge(merged_summary=("X" * 10000)) for _ in range(3)]
    mock_client = _LLMMock()
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        await rb.build_report(
            run_id="run-large",
            step14_response=_make_step14(bridges),
        )

    prompt = mock_client.prompt
    # The prompt must be bounded — three 10KB summaries should NOT result in
    # a >25KB prompt. Buggy code produces ~30KB+.
    assert len(prompt) < 25000, (
        f"Prompt was not truncated: {len(prompt)} chars (expected < 25000). "
        f"Prompt head: {prompt[:200]!r}"
    )


@pytest.mark.asyncio
async def test_build_report_user_query_not_bridge_summary():
    """SWARM-6 BUG #37: The `user_query` passed to `pack_bridges_into_context`
    must be the actual user query (a structural description), NOT
    `bridges[0].match.mechanism.source_result.merged_summary[:300]`.

    The buggy code passes the top bridge's merged_summary, which is a source
    document's summary — NOT the user's problem. This misleads the LLM."""
    from services.layer4 import report_builder as rb

    # Use a distinctive merged_summary so the test is unambiguous.
    bridge = _make_bridge(merged_summary="UNIQUE_MERGED_SUMMARY_TEXT" + "Z" * 500)
    captured: dict = {}

    def _mock_pack(bridges, user_query=""):
        captured["user_query"] = user_query
        return "context"

    mock_client = _LLMMock()
    with patch.object(rb, "pack_bridges_into_context", side_effect=_mock_pack), \
         patch.object(rb, "_get_llm_client", return_value=mock_client):
        await rb.build_report(
            run_id="run-userquery",
            step14_response=_make_step14([bridge]),
        )

    # The user_query MUST NOT be the bridge's merged_summary[:300].
    expected_bad = bridge.match.mechanism.source_result.merged_summary[:300]
    assert captured.get("user_query") != expected_bad, (
        "user_query passed to pack_bridges_into_context equals "
        "bridges[0].match.mechanism.source_result.merged_summary[:300] — "
        "the buggy code is leaking a source-document summary as the user's problem."
    )
    # Additionally, the user_query should NOT contain the merged_summary's
    # distinctive prefix.
    assert "UNIQUE_MERGED_SUMMARY_TEXT" not in (captured.get("user_query") or ""), (
        "user_query contains bridge merged_summary text — this is the bug."
    )


@pytest.mark.asyncio
async def test_build_report_output_validates_against_final_report_schema():
    """The FinalReport returned by `build_report` must pass Pydantic
    validation. A "fake fix" that returns a dict or an incomplete object
    must fail this test."""
    from services.layer4 import report_builder as rb

    mock_client = _LLMMock()
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        report = await rb.build_report(
            run_id="run-schema",
            step14_response=_make_step14([_make_bridge()]),
        )

    # Round-trip through the schema.
    dumped = report.model_dump()
    re_validated = FinalReport(**dumped)
    assert re_validated.run_id == "run-schema"
    # Each required section is a non-empty string.
    assert isinstance(re_validated.the_mechanism, str) and re_validated.the_mechanism
    assert isinstance(re_validated.the_experiment, str) and re_validated.the_experiment
    assert isinstance(re_validated.who_solved_this, str) and re_validated.who_solved_this
    assert isinstance(re_validated.next_3_actions, list)
    assert isinstance(re_validated.warnings_and_conflicts, list)


@pytest.mark.asyncio
async def test_build_report_sanitizes_api_keys_from_errors():
    """SWARM-6 BUG #23: API keys (e.g. `sk-ant-...`) must NEVER leak to the
    user via the report text or via the LLM prompt (which can echo back).

    The buggy code interpolates `merged_summary` verbatim into the prompt.
    If a source document's scraped summary happens to contain an API key,
    the key flows into the prompt and potentially into the LLM's response
    (which then becomes the user-facing report)."""
    from services.layer4 import report_builder as rb

    fake_key = "sk-ant-api03-FAKE-KEY-FOR-TESTING-1234567890"
    bridge = _make_bridge(merged_summary=f"Scraped content with key {fake_key} inside it.")

    mock_client = _LLMMock()
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        report = await rb.build_report(
            run_id="run-key-leak",
            step14_response=_make_step14([bridge]),
        )

    # The prompt sent to the LLM must NOT contain the API key.
    prompt = mock_client.prompt
    assert fake_key not in prompt, (
        f"API key leaked into LLM prompt: {prompt[:300]!r}"
    )
    # And the user-facing report text must NOT contain the API key.
    all_text = " ".join(
        [
            report.the_mechanism,
            report.the_experiment,
            report.who_solved_this,
            report.confidence_disclaimer,
            " ".join(report.next_3_actions or []),
            " ".join(report.warnings_and_conflicts or []),
        ]
    )
    assert fake_key not in all_text, (
        f"API key leaked into report text: {all_text[:300]!r}"
    )


@pytest.mark.asyncio
async def test_build_report_handles_empty_bridges():
    """`build_report` must not crash on an empty bridges list."""
    from services.layer4 import report_builder as rb

    mock_client = _LLMMock()
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        report = await rb.build_report(
            run_id="run-empty",
            step14_response=_make_step14([]),
        )

    assert isinstance(report, FinalReport)
    assert report.run_id == "run-empty"


@pytest.mark.asyncio
async def test_build_report_handles_single_bridge():
    """`build_report` must work with a single bridge (not just 2-3)."""
    from services.layer4 import report_builder as rb

    mock_client = _LLMMock()
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        report = await rb.build_report(
            run_id="run-single",
            step14_response=_make_step14([_make_bridge()]),
        )

    assert isinstance(report, FinalReport)


@pytest.mark.asyncio
async def test_build_report_handles_malformed_bridge():
    """A bridge missing required fields must raise a clear, typed error —
    NOT propagate as an unhandled AttributeError / KeyError 500.

    Note: Pydantic validates at construction time, so the test focuses on
    what `build_report` does when given a bridge whose `match.mechanism`
    fields are wrong / missing (a runtime integrity check)."""
    from services.layer4 import report_builder as rb

    # Build a valid RankedBridge, then strip out the causal_graph to simulate
    # a malformed payload (some upstream step forgot to populate it).
    bridge = _make_bridge()
    # Replace causal_graph with None — this is what "malformed" looks like
    # at runtime.
    bridge.match.mechanism.causal_graph = None  # type: ignore[assignment]

    mock_client = _LLMMock()
    # The LLM is irrelevant here — the bug is that the context-packager /
    # bridge-detail builder crashes on a malformed bridge. We just want
    # build_report to surface a clear, catchable error.
    with patch.object(rb, "_get_llm_client", return_value=mock_client):
        # Either build_report raises a clear error (ValueError / TypeError),
        # or it falls back gracefully. Either way it must NOT raise an
        # unhandled AttributeError / KeyError that would become a 500 with
        # 'NoneType' or 'KeyError' in the message.
        try:
            report = await rb.build_report(
                run_id="run-malformed",
                step14_response=_make_step14([bridge]),
            )
            # If it returns a report, fine — that's a graceful fallback.
            assert isinstance(report, FinalReport)
        except (ValueError, TypeError) as exc:
            # A clear, typed error is acceptable.
            assert (
                "bridge" in str(exc).lower()
                or "graph" in str(exc).lower()
                or "missing" in str(exc).lower()
                or "none" in str(exc).lower()
            ), (
                f"Error message should mention 'bridge'/'graph'/'missing'/'none': {exc!r}"
            )
        except Exception as exc:
            # AttributeError / KeyError / NoneType errors are NOT acceptable —
            # they indicate the malformed bridge was not detected early.
            pytest.fail(
                f"build_report raised an unhandled {type(exc).__name__} on a "
                f"malformed bridge — should be a clear ValueError/TypeError. "
                f"Original: {exc!r}"
            )