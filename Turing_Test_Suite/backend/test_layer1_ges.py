"""Tests for the GES (Greedy Equivalence Search) implementation in
``services/layer1/ges_algorithm.py``.

Each test catches a specific bug from the SWARM-3 audit (see ``worklog.md``).
Tests are independent, use deterministic fixtures, and do not require a
running LLM or DB.

Note: tests that need real causal-learn behavior use `pytest.importorskip`
and will skip if causal-learn isn't installed. The score-function test
mocks causal-learn so it runs without the dependency.
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from services.layer1 import ges_algorithm as ges_mod
from services.layer1.ges_algorithm import GESGraphBuilder


# ---------------------------------------------------------------------------
# Helper: inject a mock ges() into causal-learn's GES module
# ---------------------------------------------------------------------------
def _inject_mock_ges(monkeypatch, mock_ges):
    """Inject a mock `ges()` callable into sys.modules so the dynamic import
    inside `GESGraphBuilder.build_graph` picks it up.
    """
    ges_pkg = MagicMock()
    ges_pkg.ges = mock_ges
    # Build the parent module hierarchy so `from causallearn.search.ScoreBased.GES import ges` resolves
    monkeypatch.setitem(sys.modules, "causallearn", MagicMock())
    monkeypatch.setitem(sys.modules, "causallearn.search", MagicMock())
    monkeypatch.setitem(sys.modules, "causallearn.search.ScoreBased", MagicMock())
    monkeypatch.setitem(sys.modules, "causallearn.search.ScoreBased.GES", ges_pkg)
    return mock_ges


def _build_graph_from_adj(adj, columns):
    """Helper to construct a mock causal-learn GeneralGraph-like object."""
    return {"G": type("G", (), {"graph": adj})()}


# ===========================================================================
# GES forward phase should add correct edges (v-structure)
# ===========================================================================
def test_ges_forward_phase_adds_edges(v_structure_df):
    """Catches SWARM-3 (regression): GES forward phase should add the
    correct edges for a v-structure X → Z ← Y (collider).

    GES adds edges greedily to maximize the BIC score; for a true
    v-structure, both X→Z and Y→Z should be added.
    """
    pytest.importorskip("causallearn")
    result = GESGraphBuilder.build_graph(v_structure_df)
    edges = result["edges"]
    edge_pairs = {(e["source"], e["target"]) for e in edges}

    # X-Z and Y-Z edges should be present (collider at Z)
    assert ("X", "Z") in edge_pairs or ("Z", "X") in edge_pairs, (
        f"Forward phase failed to add X-Z edge for v-structure: {edges}"
    )
    assert ("Y", "Z") in edge_pairs or ("Z", "Y") in edge_pairs, (
        f"Forward phase failed to add Y-Z edge for v-structure: {edges}"
    )


# ===========================================================================
# GES backward phase should remove spurious edges
# ===========================================================================
def test_ges_backward_phase_removes_edges(simple_causal_df):
    """Catches SWARM-3 (regression): GES backward phase should remove
    spurious edges. In an X→Y→Z chain, X and Z are independent given Y,
    so the X-Z edge should NOT be present in the final CPDAG.
    """
    pytest.importorskip("causallearn")
    result = GESGraphBuilder.build_graph(simple_causal_df)
    edges = result["edges"]
    edge_pairs = {(e["source"], e["target"]) for e in edges}

    # In X→Y→Z chain, X-Z edge should be absent (X ⊥ Z | Y)
    assert ("X", "Z") not in edge_pairs, (
        f"BUG (regression): GES backward phase failed to remove spurious X→Z edge: {edges}"
    )
    assert ("Z", "X") not in edge_pairs, (
        f"BUG (regression): GES backward phase failed to remove spurious Z→X edge: {edges}"
    )


# ===========================================================================
# GES output must be a DAG (no cycles)
# ===========================================================================
def test_ges_does_not_create_cycles(v_structure_df):
    """Catches SWARM-3 (regression): GES output must be a DAG (no cycles).
    The greedy equivalence search algorithm guarantees a DAG output.
    """
    pytest.importorskip("causallearn")
    result = GESGraphBuilder.build_graph(v_structure_df)
    edges = result["edges"]

    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e["source"], e["target"])
    assert nx.is_directed_acyclic_graph(G), (
        f"BUG (regression): GES output has cycles: "
        f"{list(nx.simple_cycles(G))}"
    )


# ===========================================================================
# BUG #22 (related): GES score function calling convention
# ===========================================================================
def test_ges_score_function_correct(monkeypatch, simple_causal_df):
    """Catches SWARM-3 BUG #22 (related): GES score function should use
    the correct BIC formula. Verify that `ges()` is called with the data
    matrix as the first positional argument (the standard causal-learn API).

    The mock-and-inspect approach catches fake fixes that return hardcoded
    graphs without actually calling `ges()`.
    """
    # Mock causal-learn's ges() to inspect its call args
    mock_ges = MagicMock()
    mock_ges.return_value = _build_graph_from_adj(
        adj=np.array([
            [0, -1, 0],
            [1, 0, -1],
            [0, 1, 0],
        ], dtype=int),
        columns=simple_causal_df.columns.tolist(),
    )
    _inject_mock_ges(monkeypatch, mock_ges)

    result = GESGraphBuilder.build_graph(simple_causal_df)

    # ges() must have been called
    assert mock_ges.called, "BUG: GES builder did not call ges() at all"

    # First positional arg must be the data matrix (numpy array)
    args, kwargs = mock_ges.call_args
    assert len(args) >= 1, (
        f"BUG: ges() called without positional data argument: {mock_ges.call_args}"
    )
    data_arg = args[0]
    assert isinstance(data_arg, np.ndarray), (
        f"BUG: ges() data arg should be numpy.ndarray, got {type(data_arg)}"
    )
    assert data_arg.shape == simple_causal_df.shape, (
        f"BUG: ges() data shape mismatch: got {data_arg.shape}, "
        f"expected {simple_causal_df.shape}"
    )
    # Data must be numeric (float), not object dtype
    assert data_arg.dtype.kind == "f", (
        f"BUG: ges() data should be float dtype, got {data_arg.dtype}"
    )

    # Output must be a valid graph structure (not a correlation matrix)
    assert isinstance(result, dict)
    assert "nodes" in result
    assert "edges" in result


# ===========================================================================
# GES reproducibility
# ===========================================================================
def test_ges_random_seed_is_set(simple_causal_df):
    """Catches SWARM-3 (regression): GES results must be reproducible.
    Running GES twice on the same data must produce the same graph.

    GES uses a deterministic BIC score function; the result should not
    depend on global RNG state.
    """
    pytest.importorskip("causallearn")
    result1 = GESGraphBuilder.build_graph(simple_causal_df)
    result2 = GESGraphBuilder.build_graph(simple_causal_df)

    e1 = sorted([
        (e["source"], e["target"], round(e.get("confidence", 0), 4),
         round(e.get("weight", 0), 4), e.get("relation"))
        for e in result1["edges"]
    ])
    e2 = sorted([
        (e["source"], e["target"], round(e.get("confidence", 0), 4),
         round(e.get("weight", 0), 4), e.get("relation"))
        for e in result2["edges"]
    ])
    assert e1 == e2, (
        f"BUG: GES non-deterministic across runs:\n{e1}\nvs\n{e2}"
    )


# ===========================================================================
# Edge case: empty DataFrame
# ===========================================================================
def test_ges_empty_dataframe_raises_clear_error(empty_df):
    """Catches SWARM-3 (edge case): empty DataFrame should raise a clear
    ValueError, not crash with a cryptic numpy error.
    """
    with pytest.raises(ValueError, match="empty"):
        GESGraphBuilder.build_graph(empty_df)


# ===========================================================================
# BUG #22: confidence range
# ===========================================================================
def test_ges_confidence_in_zero_to_hundred_range(simple_causal_df):
    """Catches SWARM-3 BUG #22: GES edge confidence must be in [0, 100],
    not in [0, |r|] or [0, 1]. A common fake fix is to use
    `abs(correlation) * 100` without clamping, which can exceed 100 for
    perfect correlations — this test catches that.
    """
    pytest.importorskip("causallearn")
    result = GESGraphBuilder.build_graph(simple_causal_df)
    for node in result["nodes"]:
        c = node.get("confidence", 0)
        assert 0.0 <= c <= 100.0, (
            f"BUG #22: Node confidence out of [0, 100]: {c}"
        )
    for edge in result["edges"]:
        c = edge.get("confidence", 0)
        assert 0.0 <= c <= 100.0, (
            f"BUG #22: Edge confidence out of [0, 100]: {c}"
        )