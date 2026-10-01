"""Tests for the PC Algorithm implementation in
``services/layer1/pc_algorithm.py``.

Each test catches a specific bug from the SWARM-3 audit (see ``worklog.md``).
Tests are independent, use deterministic fixtures (seeded RNGs, mocked
causal-learn when needed), and do not require a running LLM or DB.
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from services.layer1 import pc_algorithm
from services.layer1.pc_algorithm import PCGraphBuilder, run_pc_algorithm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeCG:
    """Mock causal-learn CausalGraph with explicit attributes only.

    By default `ci_test` is NOT defined, so accessing it raises AttributeError
    (simulating causal-learn versions where the attribute doesn't exist).
    """

    def __init__(self, adj, include_ci_test=False, ci_test_return=0.001):
        self.G = type("G", (), {"graph": adj})()
        if include_ci_test:
            self.ci_test = MagicMock(return_value=ci_test_return)


def _inject_mock_pc(monkeypatch, mock_cg):
    """Inject a mock `pc()` callable into pc_algorithm's namespace.

    Uses `raising=False` so it works even when causal-learn isn't installed
    (in which case `pc` was never bound on the module).
    """
    def _fake_pc(*args, **kwargs):
        return mock_cg
    monkeypatch.setattr(pc_algorithm, "pc", _fake_pc, raising=False)


# ===========================================================================
# BUG #3: PC depth hardcoded to 1
# ===========================================================================
def test_pc_depth_is_not_hardcoded_to_1(cond_independent_df):
    """Catches SWARM-3 BUG #3: PC algorithm hardcoded depth=1 misses
    conditionally-independent pairs (X ⊥ Y | {Z1, Z2}).

    With depth=1, PC cannot test conditioning sets of size 2, so the X-Y
    edge is never removed even though X and Y are independent given Z1, Z2.
    A correct implementation removes the X-Y edge.
    """
    pytest.importorskip("causallearn")
    result = PCGraphBuilder.build_graph(cond_independent_df)
    edges = result["edges"]
    edge_pairs = {(e["source"], e["target"]) for e in edges}

    # X-Y edge should be absent
    assert ("X", "Y") not in edge_pairs, \
        f"BUG #3: X-Y edge present despite X ⊥ Y | {{Z1, Z2}}: {edges}"
    assert ("Y", "X") not in edge_pairs, \
        f"BUG #3: Y-X edge present despite X ⊥ Y | {{Z1, Z2}}: {edges}"

    # Fake-fix detection: a no-op (empty graph) would also pass the above.
    # Ensure PC found SOME edges (X depends on Z1, Z2; Y depends on Z1, Z2),
    # so the algorithm is actually running, not returning empty.
    assert len(edges) >= 1, "PC returned no edges — algorithm may be a no-op fake fix"
    # At least one edge should connect Z1 or Z2 to X or Y
    found_pairs = edge_pairs | {(b, a) for (a, b) in edge_pairs}
    expected_z_pairs = {
        ("Z1", "X"), ("X", "Z1"), ("Z1", "Y"), ("Y", "Z1"),
        ("Z2", "X"), ("X", "Z2"), ("Z2", "Y"), ("Y", "Z2"),
    }
    assert any(p in found_pairs for p in expected_z_pairs), \
        "PC found no Z1/Z2 → X/Y edges — algorithm may be returning empty graph"


# ===========================================================================
# BUG #12: Pearson correlation fallback fabricates causal directions
# ===========================================================================
def test_pc_no_pearson_fallback_fabricates_direction(monkeypatch):
    """Catches SWARM-3 BUG #12: PCGraphBuilder.build_graph falls back to
    Pearson correlation when PC finds no directed edges, fabricating causal
    directions from correlation signs.

    Correlation is not causation — the fallback is statistically meaningless.
    """
    # Mock pc() to return a graph with NO directed edges (forces fallback path)
    mock_cg = _FakeCG(adj=np.zeros((3, 3), dtype=int), include_ci_test=False)
    _inject_mock_pc(monkeypatch, mock_cg)

    # Strongly correlated data via a HIDDEN confounder (no direct causal link
    # between X and Y in the input data).
    rng = np.random.default_rng(42)
    n = 500
    Z_hidden = rng.normal(0, 1, n)
    X = 2.0 * Z_hidden + rng.normal(0, 0.1, n)
    Y = 3.0 * Z_hidden + rng.normal(0, 0.1, n)
    W = rng.normal(0, 1, n)  # independent column
    df = pd.DataFrame({"X": X, "Y": Y, "W": W})

    result = PCGraphBuilder.build_graph(df)
    edges = result["edges"]

    # No CORRELATES edges should be fabricated (correlation is not causation)
    correlates_edges = [e for e in edges if e.get("relation") == "CORRELATES"]
    assert len(correlates_edges) == 0, (
        f"BUG #12: PC fallback fabricated {len(correlates_edges)} CORRELATES edges: "
        f"{correlates_edges}"
    )
    # Stronger: no edges at all should be fabricated from this no-causal-edge case
    assert len(edges) == 0, (
        f"BUG #12: PC fallback fabricated {len(edges)} edges when PC found no causal edges: "
        f"{edges}"
    )


# ===========================================================================
# BUG #5 (related): PC reproducibility
# ===========================================================================
def test_pc_random_seed_is_set(simple_causal_df):
    """Catches SWARM-3 BUG #5 (related): PC results must be reproducible
    across runs. Running PC twice on the same data must produce the same graph.
    """
    pytest.importorskip("causallearn")
    result1 = PCGraphBuilder.build_graph(simple_causal_df)
    result2 = PCGraphBuilder.build_graph(simple_causal_df)
    # Compare edges with their attributes (rounded to avoid float noise)
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
        f"BUG #5: PC non-deterministic across runs:\n{e1}\nvs\n{e2}"
    )


# ===========================================================================
# Edge case: empty DataFrame
# ===========================================================================
def test_pc_empty_dataframe_raises_clear_error(empty_df):
    """Catches SWARM-3 (edge case): empty DataFrame should raise a clear
    ValueError, not crash with a cryptic KeyError or numpy error.
    """
    with pytest.raises(ValueError, match="empty"):
        PCGraphBuilder.build_graph(empty_df)


# ===========================================================================
# Edge case: single-column DataFrame
# ===========================================================================
def test_pc_single_column_raises_clear_error(monkeypatch, single_col_df):
    """Catches SWARM-3 (edge case): single-column DataFrame should raise a
    clear ValueError BEFORE calling pc(), not crash with a cryptic error
    from causal-learn (which requires >=2 columns to build a skeleton).
    """
    # Even if we mock pc to succeed, the code should reject single-column input
    mock_cg = _FakeCG(adj=np.zeros((1, 1), dtype=int), include_ci_test=False)
    _inject_mock_pc(monkeypatch, mock_cg)

    with pytest.raises(ValueError, match="column|single|at least|two|dimension"):
        PCGraphBuilder.build_graph(single_col_df)


# ===========================================================================
# Edge case: constant column
# ===========================================================================
def test_pc_handles_constant_column(constant_col_df):
    """Catches SWARM-3 (edge case): constant column should not crash the
    Fisher-Z conditional independence test (which divides by std-dev).

    PC must either skip the column, handle zero variance gracefully, or
    raise a clear ValueError — not crash with numpy/scipy errors.
    """
    pytest.importorskip("causallearn")
    try:
        result = PCGraphBuilder.build_graph(constant_col_df)
        assert "nodes" in result and "edges" in result
    except ValueError as e:
        # Acceptable: a clear error about zero-variance / constant column
        assert any(
            s in str(e).lower()
            for s in ["constant", "variance", "zero", "no variation", "degenerate"]
        ), f"PC raised unrelated ValueError: {e}"
    except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError,
            KeyError, IndexError, TypeError) as e:
        pytest.fail(
            f"BUG (edge case): PC crashed on constant column with "
            f"{type(e).__name__}: {e}"
        )


# ===========================================================================
# Edge case: perfectly collinear columns
# ===========================================================================
def test_pc_handles_perfectly_collinear_columns(collinear_df):
    """Catches SWARM-3 (edge case): perfectly collinear columns should not
    crash PC's Fisher-Z test (which inverts a singular covariance matrix).

    PC must handle singularity gracefully or raise a clear error.
    """
    pytest.importorskip("causallearn")
    try:
        result = PCGraphBuilder.build_graph(collinear_df)
        assert "nodes" in result and "edges" in result
    except ValueError as e:
        assert any(
            s in str(e).lower()
            for s in ["collinear", "singular", "rank", "variance", "degenerate"]
        ), f"PC raised unrelated ValueError: {e}"
    except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError,
            KeyError, IndexError, TypeError) as e:
        pytest.fail(
            f"BUG (edge case): PC crashed on collinear columns with "
            f"{type(e).__name__}: {e}"
        )


# ===========================================================================
# Edge case: skeleton should never contain self-loops
# ===========================================================================
def test_pc_skeleton_no_self_loops(simple_causal_df):
    """Catches SWARM-3 (edge case): PC skeleton should never contain
    self-loops (an edge from a node to itself).
    """
    pytest.importorskip("causallearn")
    result = PCGraphBuilder.build_graph(simple_causal_df)
    for e in result["edges"]:
        assert e["source"] != e["target"], \
            f"BUG (edge case): self-loop detected in PC output: {e}"


# ===========================================================================
# BUG #3 (related): v-structure orientation
# ===========================================================================
def test_pc_vstructure_orientation_correct(v_structure_df):
    """Catches SWARM-3 BUG #3 (related): given X → Z ← Y (no X-Y edge),
    PC should orient the v-structure correctly (collider at Z).
    """
    pytest.importorskip("causallearn")
    result = PCGraphBuilder.build_graph(v_structure_df)
    edges = result["edges"]
    edge_pairs = {(e["source"], e["target"]) for e in edges}

    # Both X→Z and Y→Z should be present (collider at Z)
    assert ("X", "Z") in edge_pairs or ("Z", "X") in edge_pairs, \
        f"X-Z edge missing in v-structure: {edges}"
    assert ("Y", "Z") in edge_pairs or ("Z", "Y") in edge_pairs, \
        f"Y-Z edge missing in v-structure: {edges}"
    # X-Y edge should be absent (X and Y are marginally independent)
    assert ("X", "Y") not in edge_pairs and ("Y", "X") not in edge_pairs, \
        f"BUG #3: X-Y edge present in v-structure (X,Y marginally independent): {edges}"


# ===========================================================================
# BUG #3 (related): ancestral vs collider distinction
# ===========================================================================
def test_pc_distinguishes_ancestral_from_colliders(simple_causal_df):
    """Catches SWARM-3 BUG #3 (related): PC should NOT mark an ancestral
    chain (X → Y → Z) as a v-structure (collider at Y).

    In X → Y → Z, Y is NOT a collider. If PC incorrectly orients Z→Y
    alongside X→Y, Y becomes a collider — a mis-orientation.
    """
    pytest.importorskip("causallearn")
    result = PCGraphBuilder.build_graph(simple_causal_df)
    edges = result["edges"]
    edge_pairs = {(e["source"], e["target"]) for e in edges}

    x_to_y = ("X", "Y") in edge_pairs
    z_to_y = ("Z", "Y") in edge_pairs
    # In a chain X→Y→Z, Y is not a collider — both X→Y and Z→Y should not coexist.
    assert not (x_to_y and z_to_y), (
        f"BUG #3: PC incorrectly marked Y as collider in chain X→Y→Z "
        f"(both X→Y and Z→Y present): {edges}"
    )


# ===========================================================================
# BUG #20: ci_test calling convention
# ===========================================================================
def test_pc_ci_test_called_with_correct_args(monkeypatch, simple_causal_df):
    """Catches SWARM-3 BUG #20 (calling convention): PCGraphBuilder should
    call `cg.ci_test(i, j, conditioning_set)` with integer node indices and
    a set/list of conditioning variable indices — not via an attribute
    that doesn't exist on real causal-learn CausalGraph objects.
    """
    adj = np.array([
        [0, -1, 0],
        [1, 0, -1],
        [0, 1, 0],
    ], dtype=int)
    mock_cg = _FakeCG(adj=adj, include_ci_test=True, ci_test_return=0.001)
    _inject_mock_pc(monkeypatch, mock_cg)

    PCGraphBuilder.build_graph(simple_causal_df)

    assert mock_cg.ci_test.called, "ci_test was never called"
    for call in mock_cg.ci_test.call_args_list:
        args, kwargs = call
        # First two args should be integer node indices
        assert len(args) >= 2, f"ci_test called with too few positional args: {call}"
        assert isinstance(args[0], (int, np.integer)), \
            f"ci_test first arg should be int index, got {type(args[0])}: {call}"
        assert isinstance(args[1], (int, np.integer)), \
            f"ci_test second arg should be int index, got {type(args[1])}: {call}"
        # Optional third arg should be a set/list/array of conditioning vars
        if len(args) >= 3:
            cond = args[2]
            assert isinstance(cond, (set, frozenset, list, tuple, np.ndarray, type(None))), \
                f"ci_test conditioning set should be a set/list/array, got {type(cond)}: {call}"


# ===========================================================================
# BUG #22 (related): output structure is a causal graph, not a correlation matrix
# ===========================================================================
def test_pc_returns_causal_graph_not_correlation_matrix(monkeypatch, simple_causal_df):
    """Catches SWARM-3 BUG #22 (related): PC output should be a graph
    structure (nodes with `confidence`, edges with `confidence`), not a
    raw correlation matrix.
    """
    adj = np.array([
        [0, -1, 0],
        [1, 0, -1],
        [0, 1, 0],
    ], dtype=int)
    mock_cg = _FakeCG(adj=adj, include_ci_test=True, ci_test_return=0.001)
    _inject_mock_pc(monkeypatch, mock_cg)

    result = PCGraphBuilder.build_graph(simple_causal_df)

    # Output should be a graph structure, not a correlation matrix
    assert isinstance(result, dict), f"Expected dict graph, got {type(result)}"
    assert "nodes" in result, "Missing 'nodes' key"
    assert "edges" in result, "Missing 'edges' key"
    # Each node should have a `confidence` attribute, NOT a `correlation` field
    for node in result["nodes"]:
        assert "confidence" in node, f"Node missing confidence: {node}"
        assert "correlation" not in node, \
            f"BUG #22: Node has spurious correlation field: {node}"
    # Each edge should have a `confidence` attribute
    for edge in result["edges"]:
        assert "confidence" in edge, f"Edge missing confidence: {edge}"


# ===========================================================================
# BUG #22: confidence range
# ===========================================================================
def test_pc_confidence_in_zero_to_hundred_range(monkeypatch, simple_causal_df):
    """Catches SWARM-3 BUG #22: confidence values must be in [0, 100],
    NOT in [0, 1] or [0, |r|] (the correlation scale).

    A common fake fix is to use `1 - p_value` (range [0, 1]) and forget
    to multiply by 100 — this test catches that.
    """
    adj = np.array([
        [0, -1, 0],
        [1, 0, -1],
        [0, 1, 0],
    ], dtype=int)
    mock_cg = _FakeCG(adj=adj, include_ci_test=True, ci_test_return=0.001)
    _inject_mock_pc(monkeypatch, mock_cg)

    result = PCGraphBuilder.build_graph(simple_causal_df)

    for node in result["nodes"]:
        c = node.get("confidence", 0)
        assert 0.0 <= c <= 100.0, \
            f"BUG #22: Node confidence out of [0, 100]: {c}"
    for edge in result["edges"]:
        c = edge.get("confidence", 0)
        assert 0.0 <= c <= 100.0, \
            f"BUG #22: Edge confidence out of [0, 100]: {c}"


# ===========================================================================
# BUG #20: no greedy attribute lookup / no |correlation| fallback
# ===========================================================================
def test_pc_does_not_use_greedy_attribute_lookup(monkeypatch, simple_causal_df):
    """Catches SWARM-3 BUG #20: PCGraphBuilder calls `cg.ci_test` which may
    not exist on causal-learn CausalGraph objects. When ci_test is missing,
    the code falls back to `confidence = abs(correlation)` — which is
    statistically meaningless (correlation is not causal confidence).
    """
    # Mock pc() with a directed X→Y edge but NO ci_test attribute
    adj = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=int)
    # include_ci_test=False → accessing mock_cg.ci_test raises AttributeError
    mock_cg = _FakeCG(adj=adj, include_ci_test=False)
    _inject_mock_pc(monkeypatch, mock_cg)

    result = PCGraphBuilder.build_graph(simple_causal_df)
    edges = result["edges"]

    # X→Y edge should be present (it's in the adj matrix)
    x_to_y = [e for e in edges if e["source"] == "X" and e["target"] == "Y"]
    assert len(x_to_y) == 1, f"Expected exactly one X→Y edge, got: {edges}"
    edge = x_to_y[0]

    # Compute the |correlation| * 100 that the buggy fallback would produce.
    # NOTE: the code does `confidence = abs(correlation)` then
    # `confidence * 100.0` (NO rounding), so edge["confidence"] exactly
    # equals `corr * 100.0` when the fallback fires.
    data = simple_causal_df.to_numpy()
    corr = abs(float(np.corrcoef(data[:, 0], data[:, 1])[0, 1]))
    buggy_confidence = corr * 100.0

    # Edge confidence should NOT approximately equal |correlation| * 100.
    # We use a tolerance of 0.5 (the fix's p_value-based confidence will
    # differ from |r|*100 by more than 0.5 for typical data).
    assert abs(edge["confidence"] - buggy_confidence) > 0.5, (
        f"BUG #20: edge confidence {edge['confidence']} ≈ |r| * 100 = "
        f"{buggy_confidence} — fallback to |correlation| is statistically "
        f"meaningless (correlation is not causal confidence)"
    )