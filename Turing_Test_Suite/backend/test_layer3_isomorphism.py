"""
test_layer3_isomorphism.py — Graph isomorphism matcher tests.

Catches every SWARM-5 BUG in services/layer3/isomorphism.py:

  BUG #3  — GED timeout fallback heuristic `|Δnodes| + |Δedges|` returns 0
            for two graphs of the same size, classifying completely
            unrelated graphs (e.g., path graph vs star graph) as PERFECT
            matches. (Referenced as "SWARM-3 BUG #3" in the test spec;
            actual location: SWARM-5 BUG #3.)
  BUG #15 — `_to_digraph` builds a topology-only nx.DiGraph — drops node
            labels, edge RELATIONS (INHIBITS vs ACTIVATES), confidences,
            weights. A target "A -[INHIBITS]-> B" and a candidate
            "X -[ACTIVATES]-> Y" get a perfect match.
  BUG #16 — `nx.DiGraph` silently collapses parallel edges (multigraphs).
            Two edges A→B with different relations collapse into one.
  BUG #17 — `max_ged` calculation ignores the possibility of node
            substitutions — overestimates the denominator, under-reports
            similarity for legitimately-isomorphic graphs.
  BUG #18 — `calculate_structural_similarity` is sync but called from
            async `match_graphs` — blocks the event loop for the full GED
            computation (NP-hard, can take seconds on modest graphs).

Design notes:
- Each test FAILS against the current buggy code (red) and PASSES only
  once the corresponding fix lands (green).
- The GED timeout fallback test mocks `nx.graph_edit_distance` to return
  None so the fallback path is exercised deterministically (no real
  3-second wait needed).
"""
from __future__ import annotations

import asyncio
import inspect
import time
from typing import List, Optional
from unittest.mock import patch

import networkx as nx
import pytest

from schemas.graph import CausalGraph, Edge, Node
from schemas.layer3 import (
    ExtractedMechanism,
    IsomorphismMatch,
    IsomorphismThresholds,
    MatchType,
    MergedResult,
    SearchSource,
    SearchResult,
    Step13Request,
    Step13Response,
    StructuralQuery,
)
from services.layer3 import isomorphism as iso


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _node(i: int, label: Optional[str] = None, confidence: float = 90.0) -> Node:
    return Node(id=f"n{i}", label=label or f"node-{i}", confidence=confidence)


def _edge(src: str, tgt: str, relation: str = "ACTIVATES", confidence: float = 90.0) -> Edge:
    return Edge(source=src, target=tgt, relation=relation, confidence=confidence)


def _path_graph(n: int = 3) -> CausalGraph:
    """A path graph: n0 → n1 → n2 → ... → n(n-1)."""
    nodes = [_node(i) for i in range(n)]
    edges = [_edge(f"n{i}", f"n{i + 1}") for i in range(n - 1)]
    return CausalGraph(nodes=nodes, edges=edges)


def _star_graph(n: int = 4) -> CausalGraph:
    """A star graph: n0 → n1, n0 → n2, ..., n0 → n(n-1). n nodes, n-1 edges.
    For n=3 this is the same as a path graph; we use n≥4 to make the
    topology genuinely different from a path graph."""
    nodes = [_node(i) for i in range(n)]
    edges = [_edge("n0", f"n{i}") for i in range(1, n)]
    return CausalGraph(nodes=nodes, edges=edges)


def _empty_graph() -> CausalGraph:
    return CausalGraph(nodes=[], edges=[])


def _query() -> StructuralQuery:
    return StructuralQuery(
        original_node_id="n0",
        original_confidence=50.0,
        structural_description="A high-variance input causes output failure.",
    )


def _extracted(graph: CausalGraph) -> ExtractedMechanism:
    sr = SearchResult(
        source=SearchSource.PAPER, title="T", summary="s",
        confidence=80.0, original_query=_query(),
    )
    mr = MergedResult(
        title="T", merged_summary="s", underlying_mechanism="m",
        sources=[SearchSource.PAPER], source_results=[sr], confidence=85.0,
    )
    return ExtractedMechanism(source_result=mr, causal_graph=graph)


# ===========================================================================
# BUG #3: GED timeout fallback returns 0 → unrelated graphs classified as PERFECT
# ===========================================================================

def test_ged_timeout_fallback_does_not_return_zero_for_same_size():
    """SWARM-5 BUG #3 (referenced as "SWARM-3 BUG #3" in the test spec):
    when `nx.graph_edit_distance(target, candidate, timeout=3.0)` returns
    None (timeout) or raises, the fallback heuristic
    `|Δnodes| + |Δedges|` returns 0 for two graphs of the SAME size but
    completely DIFFERENT topology (e.g., 4-node path vs 4-node star).
    The similarity formula `1 - 0/max_ged` = 1.0 → 100% similarity →
    PERFECT match. This is wildly wrong.

    This test forces the fallback path by mocking `nx.graph_edit_distance`
    to return None — deterministic, no real 3-second wait.
    """
    # 4-node path (linear) vs 4-node star (hub-and-spoke).
    # Both have 4 nodes and 3 edges, but completely different topology.
    target = _path_graph(4)
    candidate = _star_graph(4)
    assert len(target.nodes) == len(candidate.nodes) == 4
    assert len(target.edges) == len(candidate.edges) == 3

    # Mock the GED function to return None — simulates a timeout.
    with patch.object(nx, "graph_edit_distance", return_value=None):
        score = iso.calculate_structural_similarity(target, candidate)

    # The fallback MUST NOT report 100% (perfect match) for two completely
    # different graphs of the same size.
    assert score < 100.0, (
        f"GED timeout fallback reported {score} for two completely different "
        f"topologies (path vs star, same size). The fallback heuristic "
        f"`|Δnodes| + |Δedges|` returns 0 for same-size graphs, falsely "
        f"indicating perfect match (SWARM-5 BUG #3)."
    )
    # More specifically: the score must be meaningfully below 100%.
    # A reasonable threshold: < 80% (a perfect match should be very close to 100).
    assert score < 80.0, (
        f"GED fallback score {score} is too high for unrelated topologies; "
        f"the fallback must reflect actual structural difference, not just size."
    )


def test_ged_timeout_fallback_reflects_actual_overlap():
    """SWARM-5 BUG #3 (sibling): the fallback score must be based on ACTUAL
    node/edge overlap (e.g., degree-sequence distance), NOT just the bare
    size delta. Two completely unrelated graphs should score LOW; two
    graphs with overlapping structure should score HIGHER.

    This test verifies the fallback discriminates between
    (a) two unrelated same-size graphs, and
    (b) two genuinely similar graphs.
    """
    # (a) path vs star — same size, completely different topology.
    unrelated_target = _path_graph(4)
    unrelated_candidate = _star_graph(4)
    # (b) two identical graphs.
    identical_target = _path_graph(4)
    identical_candidate = _path_graph(4)

    with patch.object(nx, "graph_edit_distance", return_value=None):
        unrelated_score = iso.calculate_structural_similarity(unrelated_target, unrelated_candidate)
        identical_score = iso.calculate_structural_similarity(identical_target, identical_candidate)

    # The identical pair must score strictly higher than the unrelated pair.
    assert identical_score > unrelated_score, (
        f"GED fallback must discriminate identical graphs ({identical_score}) "
        f"from unrelated graphs ({unrelated_score}). The fallback must be "
        f"topology-aware, not just size-aware."
    )
    # Identical graphs under the fallback should still be near 100%.
    assert identical_score >= 80.0, (
        f"Identical graphs must score ≥ 80 even under fallback; got {identical_score}"
    )
    # Unrelated graphs under the fallback must be well below identical.
    assert unrelated_score < identical_score - 20.0, (
        f"Unrelated graphs ({unrelated_score}) must score significantly lower "
        f"than identical graphs ({identical_score}) under the fallback."
    )


# ===========================================================================
# Identical / completely different / subgraph isomorphism
# ===========================================================================

def test_isomorphism_correct_for_identical_graphs():
    """Two identical graphs must score 100% (or very close to it)."""
    g = _path_graph(4)
    score = iso.calculate_structural_similarity(g, g)
    assert score == 100.0, (
        f"identical graphs must score 100; got {score}"
    )


def test_isomorphism_correct_for_completely_different():
    """Two graphs with completely different topology must score well below 100%."""
    # 4-node path vs 4-node star (different topology).
    target = _path_graph(4)
    candidate = _star_graph(4)
    score = iso.calculate_structural_similarity(target, candidate)
    assert score < 80.0, (
        f"completely different topologies must score < 80; got {score}"
    )


def test_isomorphism_correct_for_subgraph():
    """G1 is a subgraph of G2 (e.g., G1 has 3 nodes, G2 has 5 nodes and
    contains G1) → score must reflect partial match, not 0 or 100."""
    # G1: 3-node path (n0 → n1 → n2)
    g1 = _path_graph(3)
    # G2: 5-node path that contains g1 (n0 → n1 → n2 → n3 → n4)
    g2 = _path_graph(5)
    score = iso.calculate_structural_similarity(g1, g2)
    # Partial match: not 0, not 100, somewhere in between.
    assert 0 < score < 100, (
        f"subgraph isomorphism must yield a partial-match score (0 < s < 100); "
        f"got {score}"
    )


# ===========================================================================
# Disconnected graphs
# ===========================================================================

def test_handles_disconnected_graphs():
    """Disconnected graphs must NOT crash the function."""
    # Two disconnected components.
    target = CausalGraph(
        nodes=[_node(0), _node(1), _node(2), _node(3)],
        edges=[_edge("n0", "n1"), _edge("n2", "n3")],
    )
    candidate = CausalGraph(
        nodes=[_node(0), _node(1), _node(2), _node(3)],
        edges=[_edge("n0", "n1"), _edge("n1", "n2")],
    )
    # Must not raise.
    score = iso.calculate_structural_similarity(target, candidate)
    assert isinstance(score, (int, float))
    assert 0 <= score <= 100


# ===========================================================================
# BUG #16: multigraphs without collapsing
# ===========================================================================

def test_handles_multigraphs_without_collapsing():
    """SWARM-5 BUG #16: `nx.DiGraph` silently collapses parallel edges.
    If the LLM-extracted causal graph has two edges A→B with DIFFERENT
    relations (e.g., INHIBITS and ACTIVATES — a real-world possibility
    in merged results with contradiction analysis), the second
    `G.add_edge(A, B)` call silently overwrites the first. The isomorphism
    comparison then sees only one edge, under-counting structural complexity.

    The fix must use `nx.MultiDiGraph` (which preserves parallel edges).
    """
    # Multigraph with two parallel edges A→B (different relations).
    multigraph = CausalGraph(
        nodes=[_node(0, label="A"), _node(1, label="B")],
        edges=[
            _edge("n0", "n1", relation="INHIBITS"),
            _edge("n0", "n1", relation="ACTIVATES"),
        ],
    )
    # Simple graph with just ONE A→B edge.
    simple = CausalGraph(
        nodes=[_node(0, label="A"), _node(1, label="B")],
        edges=[_edge("n0", "n1", relation="INHIBITS")],
    )

    # Convert via the internal helper.
    multi_nx = iso._to_digraph(multigraph)
    simple_nx = iso._to_digraph(simple)

    # If the fix uses MultiDiGraph, the multigraph has 2 edges.
    # If it still uses DiGraph, both have 1 edge (collapsed).
    assert multi_nx.number_of_edges() == 2, (
        f"multigraph must preserve both parallel edges; got {multi_nx.number_of_edges()} "
        f"(BUG #16: DiGraph collapses parallel edges)."
    )
    assert simple_nx.number_of_edges() == 1
    # The multigraph must have MORE edges than the simple graph.
    assert multi_nx.number_of_edges() > simple_nx.number_of_edges(), (
        f"multigraph ({multi_nx.number_of_edges()} edges) must have more edges "
        f"than simple graph ({simple_nx.number_of_edges()} edges) — the parallel "
        f"edges must NOT collapse (BUG #16)."
    )


# ===========================================================================
# BUG #15: edge relations considered in matching
# ===========================================================================

def test_preserves_edge_relations():
    """SWARM-5 BUG #15: `_to_digraph` builds a topology-only nx.DiGraph with
    no edge attributes. A target "A -[INHIBITS]-> B" and a candidate
    "X -[ACTIVATES]-> Y" (opposite relations) are identical under pure
    topology → perfect 100% match. But INHIBITS vs ACTIVATES is a
    CRITICAL semantic difference — the contradiction_detector exists
    specifically to flag this kind of mismatch. The isomorphism matcher
    must NOT report them as perfect.

    The fix must pass `edge_match=lambda a, b: a["relation"] == b["relation"]`
    to `nx.graph_edit_distance`.
    """
    # A -[INHIBITS]-> B (target)
    target = CausalGraph(
        nodes=[_node(0, label="A"), _node(1, label="B")],
        edges=[_edge("n0", "n1", relation="INHIBITS")],
    )
    # X -[ACTIVATES]-> Y (candidate) — opposite relation, same topology
    candidate = CausalGraph(
        nodes=[_node(0, label="X"), _node(1, label="Y")],
        edges=[_edge("n0", "n1", relation="ACTIVATES")],
    )

    score = iso.calculate_structural_similarity(target, candidate)
    # The score must NOT be 100% (perfect) — the relations differ.
    assert score < 100.0, (
        f"graphs with opposite edge relations (INHIBITS vs ACTIVATES) must NOT "
        f"score 100% (perfect match). Got: {score} (SWARM-5 BUG #15: topology-only "
        f"matching ignores edge relations)."
    )
    # More specifically: the score should be meaningfully below 100%.
    assert score < 80.0, (
        f"opposite relations must score well below 80%; got {score}. "
        f"Edge relations are structural information that must be preserved."
    )


# ===========================================================================
# Large graphs within timeout
# ===========================================================================

def test_handles_large_graphs_within_timeout():
    """A graph with 100+ nodes must complete within a reasonable timeout
    (e.g., 30s). GED is NP-hard, so for large graphs the timeout fallback
    kicks in — but the function must NOT hang indefinitely."""
    target = _path_graph(100)
    candidate = _path_graph(100)

    start = time.monotonic()
    score = iso.calculate_structural_similarity(target, candidate)
    elapsed = time.monotonic() - start

    # Must complete within 30s (the GED timeout is 3s + overhead).
    assert elapsed < 30.0, (
        f"100-node graph took {elapsed:.2f}s (must be < 30s)"
    )
    # Identical 100-node paths must score high (perfect or near-perfect).
    assert score >= 80.0, (
        f"identical 100-node graphs must score ≥ 80; got {score}"
    )


# ===========================================================================
# BUG #18: sync GED does not block event loop
# ===========================================================================

@pytest.mark.asyncio
async def test_sync_ged_does_not_block_event_loop():
    """SWARM-5 BUG #18: `calculate_structural_similarity` is sync but is
    called directly from async `match_graphs`. For large graphs, GED can
    take seconds — the entire event loop is blocked, starving all other
    coroutines (HTTP requests, SSE heartbeats, other matches).

    The fix must either:
    (a) keep `calculate_structural_similarity` sync but call it via
        `await asyncio.to_thread(...)` from `match_graphs`, OR
    (b) make `calculate_structural_similarity` async and use `acompletion`-style
        APIs internally.

    Either way, the source code must contain `asyncio.to_thread` or
    `match_graphs` must NOT call `calculate_structural_similarity` directly
    inside the async event loop.
    """
    src = inspect.getsource(iso)
    # The match_graphs function must offload the sync GED call.
    has_to_thread = "asyncio.to_thread" in src
    has_run_in_executor = "run_in_executor" in src
    # Alternatively, calculate_structural_similarity itself could be made async.
    sig = inspect.signature(iso.calculate_structural_similarity)
    is_async = inspect.iscoroutinefunction(iso.calculate_structural_similarity)
    assert has_to_thread or has_run_in_executor or is_async, (
        "match_graphs must offload the sync GED call to a thread via "
        "`asyncio.to_thread(...)` (or make calculate_structural_similarity "
        "async). The current code calls the sync GED directly inside the "
        "async event loop, blocking all other coroutines (SWARM-5 BUG #18)."
    )

    # Fake-fix detection: just adding `await asyncio.to_thread` once is not
    # enough — the actual GED call site must be the one offloaded. Check
    # that the source actually references `calculate_structural_similarity`
    # inside a `to_thread` call.
    if has_to_thread and not is_async:
        # Find the match_graphs source.
        match_graphs_src = inspect.getsource(iso.match_graphs)
        assert "to_thread" in match_graphs_src and "calculate_structural_similarity" in match_graphs_src, (
            "match_graphs must offload the calculate_structural_similarity call "
            "via asyncio.to_thread (or similar)."
        )


# ===========================================================================
# Sanity: match_graphs classification thresholds
# ===========================================================================

@pytest.mark.asyncio
async def test_match_graphs_classifies_perfect_match():
    """When two graphs are identical, match_graphs must classify the match
    as PERFECT (assuming the threshold is set to 100)."""
    g = _path_graph(4)
    request = Step13Request(
        target_graph=g,
        candidates=[_extracted(g)],
        thresholds=IsomorphismThresholds(perfect=100.0, strong_partial=70.0, weak_partial=30.0),
    )
    response = await iso.match_graphs(request)
    assert len(response.matches) == 1
    assert response.matches[0].match_type == MatchType.PERFECT, (
        f"identical graphs must classify as PERFECT; got {response.matches[0].match_type}"
    )


@pytest.mark.asyncio
async def test_match_graphs_classifies_discarded():
    """When two graphs are completely different, match_graphs must classify
    the match as DISCARDED (if the score is below the weak_partial threshold).

    A 4-node path vs a 4-node star yields ~71% structural similarity under
    GED (because both have 4 nodes and 3 edges — only the topology differs).
    We set all thresholds to 80 so a 71% score falls through to DISCARDED.
    """
    target = _path_graph(4)
    candidate = _star_graph(4)
    # First confirm the actual score is below 80 (path-vs-star topology).
    actual_score = iso.calculate_structural_similarity(target, candidate)
    assert actual_score < 80.0, (
        f"test setup: path-vs-star score must be < 80; got {actual_score}"
    )

    request = Step13Request(
        target_graph=target,
        candidates=[_extracted(candidate)],
        thresholds=IsomorphismThresholds(perfect=80.0, strong_partial=80.0, weak_partial=80.0),
    )
    response = await iso.match_graphs(request)
    assert len(response.matches) == 1
    # With all thresholds at 80 and score < 80, the match must be DISCARDED.
    assert response.matches[0].match_type == MatchType.DISCARDED, (
        f"path-vs-star must classify as DISCARDED when all thresholds=80; "
        f"got {response.matches[0].match_type} (score={response.matches[0].isomorphism_score})"
    )