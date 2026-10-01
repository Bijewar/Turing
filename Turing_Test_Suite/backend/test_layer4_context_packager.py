r"""
test_layer4_context_packager.py — Layer 4 Context Packager tests.

Catches every SWARM-6 BUG in services/layer4/context_packager.py:
  BUG #38 — `merged_summary` is not truncated → can blow context window
  BUG #39 — nodes/edges slicing `[:5]` doesn't indicate "(+N more)"

Design notes:
- All tests are deterministic and synchronous (the packager is a pure function).
- Tests FAIL on the current buggy code (red) and PASS only when the bug is
  properly fixed (green). Fake-fix detection assertions are included where
  relevant (e.g. truncation must happen at word boundary AND show ellipsis,
  not just be a hard slice).
"""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from schemas.layer3 import (
    ContradictionDetails,
    DebateTranscript,
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
)
from schemas.graph import CausalGraph, Edge, Node


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _make_node(i: int, label: Optional[str] = None) -> Node:
    return Node(id=f"n{i}", label=label or f"node-{i}", confidence=85.0)


def _make_edge(i: int, src: str, tgt: str, relation: str = "ACTIVATES") -> Edge:
    return Edge(source=src, target=tgt, relation=relation, confidence=80.0)


def _make_graph(n_nodes: int = 10, n_edges: int = 10) -> CausalGraph:
    nodes = [_make_node(i) for i in range(n_nodes)]
    edges = [
        _make_edge(i, f"n{i % n_nodes}", f"n{(i + 1) % n_nodes}")
        for i in range(n_edges)
    ]
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
        title="Some paper",
        summary="A deployed production system.",
        url="https://example.com",
        confidence=80.0,
        original_query=_make_structural_query(),
        citation_count=10,
        replication_count=2,
        deployment_status="deployed",
    )


def _make_merged_result(
    title: str = "Some Title",
    merged_summary: Optional[str] = None,
    underlying_mechanism: str = "Bottleneck constriction mechanism.",
) -> MergedResult:
    return MergedResult(
        title=title,
        merged_summary=merged_summary if merged_summary is not None else "Some merged summary text." * 3,
        underlying_mechanism=underlying_mechanism,
        sources=[SearchSource.WEB],
        source_results=[_make_search_result()],
        confidence=82.0,
        contradiction_analysis=None,
    )


def _make_extracted_mechanism(
    merged_summary: Optional[str] = None,
    n_nodes: int = 10,
    n_edges: int = 10,
) -> ExtractedMechanism:
    return ExtractedMechanism(
        source_result=_make_merged_result(merged_summary=merged_summary),
        causal_graph=_make_graph(n_nodes=n_nodes, n_edges=n_edges),
    )


def _make_isomorphism_match(
    isomorphism_score: float = 87.5,
    merged_summary: Optional[str] = None,
    n_nodes: int = 10,
    n_edges: int = 10,
) -> IsomorphismMatch:
    return IsomorphismMatch(
        mechanism=_make_extracted_mechanism(
            merged_summary=merged_summary,
            n_nodes=n_nodes,
            n_edges=n_edges,
        ),
        isomorphism_score=isomorphism_score,
        match_type=MatchType.STRONG_PARTIAL,
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
    merged_summary: Optional[str] = None,
    n_nodes: int = 10,
    n_edges: int = 10,
) -> RankedBridge:
    return RankedBridge(
        match=_make_isomorphism_match(
            merged_summary=merged_summary,
            n_nodes=n_nodes,
            n_edges=n_edges,
        ),
        scores=_make_validity_scores(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_pack_bridges_truncates_merged_summary():
    """SWARM-6 BUG #38: `pack_bridges_into_context` must cap `merged_summary`
    (e.g. to 800 chars) so a 10KB Wikipedia summary doesn't blow the LLM
    context window. The buggy code includes the full `merged_summary`
    verbatim."""
    from services.layer4.context_packager import pack_bridges_into_context

    huge_summary = "W" * 10000  # 10KB
    bridge = _make_bridge(merged_summary=huge_summary)

    out = pack_bridges_into_context(bridges=[bridge], user_query="some query")

    # The huge_summary itself must not appear verbatim; the summary section
    # should be bounded.
    assert huge_summary not in out, (
        "Full 10KB merged_summary leaked into the packed context — must be truncated."
    )

    # Find the "Summary:" line and check its length is bounded.
    summary_line = None
    for line in out.splitlines():
        if line.strip().startswith("Summary:"):
            summary_line = line.strip()
            break
    assert summary_line is not None, "Expected a 'Summary:' line in the packed context."
    # The summary value must be < 2000 chars (we expect ≤ 800 chars + ellipsis).
    summary_value = summary_line[len("Summary:"):].strip()
    assert len(summary_value) < 2000, (
        f"Truncated summary is {len(summary_value)} chars — must be ≤ ~800. "
        f"Got: {summary_value[:100]!r}..."
    )


def test_pack_bridges_truncation_at_word_boundary():
    """SWARM-6 BUG #38 (word boundary): Truncation must happen at a word
    boundary, not in the middle of a word. A "fake fix" that just slices
    `[:800]` would cut mid-word.

    This is a "fake-fix detection" test:
    - Buggy code: no truncation → test passes trivially (no word boundary
      check needed).
    - Fake fix (hard slice): truncates mid-word → test FAILS.
    - Proper fix (truncate at word boundary + ellipsis): test passes."""
    from services.layer4.context_packager import pack_bridges_into_context

    # Use a long summary whose 800th char is in the middle of a word.
    word = "abcdefghij" * 10  # 100-char "word" (no spaces)
    sentence = word + " "
    huge_summary = sentence * 100  # 10100 chars, each "word" is 100 chars
    bridge = _make_bridge(merged_summary=huge_summary)

    out = pack_bridges_into_context(bridges=[bridge], user_query="q")

    # Find the summary value.
    summary_value = None
    for line in out.splitlines():
        if line.strip().startswith("Summary:"):
            summary_value = line.strip()[len("Summary:"):].strip()
            break
    assert summary_value is not None, "Expected a 'Summary:' line in the packed context."

    # Only check word-boundary semantics if truncation actually happened
    # (i.e. the summary is shorter than the input).
    if len(summary_value) < len(huge_summary):
        # Accept either Unicode ellipsis … or triple-dot ...
        assert summary_value.endswith("…") or summary_value.endswith("..."), (
            f"Truncated summary doesn't end with an ellipsis indicator: {summary_value[-30:]!r}"
        )
        # Get the truncated content (without ellipsis).
        if summary_value.endswith("…"):
            truncated = summary_value[:-1]
        else:
            truncated = summary_value[:-3]

        # The truncated content must NOT end mid-word. Check that the
        # next char in the original (at the truncation position) is a word
        # boundary (space / punctuation / newline). If it's a letter, the
        # truncation cut a word in the middle.
        truncated_stripped_len = len(truncated.rstrip())
        if truncated_stripped_len < len(huge_summary):
            next_char = huge_summary[truncated_stripped_len]
            assert next_char in (" ", ",", ".", ";", ":", "!", "\n", "\t"), (
                f"Truncation cut mid-word at position {truncated_stripped_len}; "
                f"next char in original is {next_char!r}. "
                f"Truncated tail: {summary_value[-80:]!r}"
            )


def test_pack_bridges_shows_ellipsis_when_truncated():
    """SWARM-6 BUG #38 (ellipsis): When `merged_summary` is truncated, the
    packed output must end the summary with an ellipsis (… or ...) to
    indicate to the LLM (and the user) that content was elided."""
    from services.layer4.context_packager import pack_bridges_into_context

    huge_summary = "This is a sentence. " * 500  # ~10KB
    bridge = _make_bridge(merged_summary=huge_summary)

    out = pack_bridges_into_context(bridges=[bridge], user_query="q")

    summary_value = None
    for line in out.splitlines():
        if line.strip().startswith("Summary:"):
            summary_value = line.strip()[len("Summary:"):].strip()
            break
    assert summary_value is not None, "Expected a 'Summary:' line in the packed context."

    # If the input was longer than the output, an ellipsis must be shown.
    if len(summary_value) < len(huge_summary):
        assert summary_value.endswith("…") or summary_value.endswith("..."), (
            f"Truncated summary must end with ellipsis; got: {summary_value[-50:]!r}"
        )


def test_pack_bridges_node_summary_indicates_more():
    """SWARM-6 BUG #39: When the causal graph has more than 5 nodes, the
    packed output must indicate "(+N more)" so the LLM doesn't believe the
    graph has only 5 nodes when it has 20."""
    from services.layer4.context_packager import pack_bridges_into_context

    # 20 nodes, 10 edges.
    bridge = _make_bridge(n_nodes=20, n_edges=10)
    out = pack_bridges_into_context(bridges=[bridge], user_query="q")

    # Find the "Causal nodes:" line.
    nodes_line = None
    for line in out.splitlines():
        if line.strip().startswith("Causal nodes:"):
            nodes_line = line.strip()
            break
    assert nodes_line is not None, "Expected a 'Causal nodes:' line in the packed context."

    # The line must indicate that 15 more nodes exist beyond the first 5.
    # Look for "(+15 more)" or similar "(+N more)" pattern.
    import re
    more_pattern = re.search(r"\(\+(\d+)\s*more\)", nodes_line)
    assert more_pattern, (
        f"Causal nodes line does not indicate (+N more) for 20-node graph: {nodes_line!r}"
    )
    n_more = int(more_pattern.group(1))
    assert n_more == 15, (
        f"Expected (+15 more) for 20-node graph (5 shown), got (+{n_more} more): {nodes_line!r}"
    )


def test_pack_bridges_edge_summary_indicates_more():
    """SWARM-6 BUG #39: Same as nodes, but for edges — when more than 5 edges
    exist, the packed output must indicate "(+N more)"."""
    from services.layer4.context_packager import pack_bridges_into_context

    # 5 nodes, 20 edges.
    bridge = _make_bridge(n_nodes=5, n_edges=20)
    out = pack_bridges_into_context(bridges=[bridge], user_query="q")

    edges_line = None
    for line in out.splitlines():
        if line.strip().startswith("Causal edges:"):
            edges_line = line.strip()
            break
    assert edges_line is not None, "Expected a 'Causal edges:' line in the packed context."

    import re
    more_pattern = re.search(r"\(\+(\d+)\s*more\)", edges_line)
    assert more_pattern, (
        f"Causal edges line does not indicate (+N more) for 20-edge graph: {edges_line!r}"
    )
    n_more = int(more_pattern.group(1))
    assert n_more == 15, (
        f"Expected (+15 more) for 20-edge graph (5 shown), got (+{n_more} more): {edges_line!r}"
    )


def test_pack_bridges_handles_empty_bridges():
    """`pack_bridges_into_context` must not crash on an empty bridges list.
    It should return a sensible empty/header string."""
    from services.layer4.context_packager import pack_bridges_into_context

    out = pack_bridges_into_context(bridges=[], user_query="q")
    assert isinstance(out, str)
    # Must not raise. May contain a header like "TOP 0 CROSS-DOMAIN BRIDGES FOUND".
    # The only requirement is that it's a string, not a crash.


def test_pack_bridges_handles_single_bridge():
    """`pack_bridges_into_context` must work with a single bridge."""
    from services.layer4.context_packager import pack_bridges_into_context

    bridge = _make_bridge(n_nodes=3, n_edges=3)
    out = pack_bridges_into_context(bridges=[bridge], user_query="q")
    assert isinstance(out, str)
    assert "BRIDGE #1" in out, f"Expected 'BRIDGE #1' in output; got: {out[:200]!r}"


def test_pack_bridges_does_not_leak_secrets():
    """If any bridge field accidentally contains an API key (e.g. from
    scraped web content), the packed context must redact it. The buggy code
    interpolates `merged_summary` verbatim, so a key would leak through."""
    from services.layer4.context_packager import pack_bridges_into_context

    fake_key = "sk-ant-api03-FAKE-KEY-FOR-TESTING-1234567890"
    bridge = _make_bridge(merged_summary=f"Scraped content with key {fake_key} inside.")

    out = pack_bridges_into_context(bridges=[bridge], user_query="q")
    assert fake_key not in out, (
        f"API key leaked into packed context: {out[:300]!r}"
    )


def test_pack_bridges_total_length_within_token_limit():
    """The total packed output must be bounded so it doesn't blow the LLM
    token budget. Three bridges each with a 10KB summary + 20-node graph
    should NOT produce a 30KB+ packed context."""
    from services.layer4.context_packager import pack_bridges_into_context

    bridges = [
        _make_bridge(merged_summary="Y" * 10000, n_nodes=20, n_edges=20)
        for _ in range(3)
    ]
    out = pack_bridges_into_context(bridges=bridges, user_query="q")

    # 3 bridges × 10KB summaries × 20 nodes/edges each, if not truncated, is
    # ~30KB+. After truncation, must be well under 10KB.
    # Allow some slack for headers / scores.
    assert len(out) < 15000, (
        f"Total packed context is {len(out)} chars — must be < 15000 to fit "
        f"in a typical LLM context window. Head: {out[:200]!r}"
    )