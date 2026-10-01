"""
Pydantic schema tests for `schemas/graph.py`.

Catches:
- SWARM-6 BUG #57 — `CausalGraph` has no model_validator for dangling
  edges, duplicate node IDs, or self-loops.
- SWARM-6 BUG #58 — `Edge.weight` has no `ge`/`le` constraint — accepts
  NaN, infinity, out-of-range.
- SWARM-6 BUG #59 — `Edge.relation` is a bare `str` — no enum, no
  normalization (case/whitespace).

Every test below FAILS against the current buggy code and PASSES only
once the corresponding fix lands.
"""
import math

import pytest
from pydantic import ValidationError


from schemas.graph import CausalGraph, Edge, Node


def _valid_node(id="a", label="A", confidence=50.0):
    """Helper — builds a fully-populated Node."""
    return Node(id=id, label=label, confidence=confidence)


def _valid_edge(source="a", target="b", relation="CAUSES", confidence=50.0, weight=1.0):
    """Helper — builds a fully-populated Edge."""
    return Edge(source=source, target=target, relation=relation,
                confidence=confidence, weight=weight)


# ===========================================================================
# CausalGraph structural integrity (SWARM-6 BUG #57).
# ===========================================================================

def test_causal_graph_no_duplicate_node_ids():
    """SWARM-6 BUG #57: duplicate node IDs MUST raise ValidationError.

    Two nodes with the same `id` are ambiguous — downstream isomorphism
    and graph rendering break. The fix adds a `model_validator` that
    rejects duplicates.
    """
    with pytest.raises(ValidationError):
        CausalGraph(
            nodes=[_valid_node(id="a"), _valid_node(id="a")],
            edges=[],
        )


def test_causal_graph_no_dangling_edge_source():
    """SWARM-6 BUG #57: edge whose `source` is not in `nodes` MUST raise.

    Dangling edges corrupt NOTEARS fitting, the UI canvas, and Step 13
    isomorphism matching. The fix rejects them at the schema boundary.
    """
    with pytest.raises(ValidationError):
        CausalGraph(
            nodes=[_valid_node(id="a")],
            edges=[Edge(source="ghost", target="a", relation="CAUSES", confidence=50.0)],
        )


def test_causal_graph_no_dangling_edge_target():
    """SWARM-6 BUG #57: edge whose `target` is not in `nodes` MUST raise."""
    with pytest.raises(ValidationError):
        CausalGraph(
            nodes=[_valid_node(id="a")],
            edges=[Edge(source="a", target="ghost", relation="CAUSES", confidence=50.0)],
        )


def test_causal_graph_no_self_loops():
    """SWARM-6 BUG #57: self-loop edges (source==target) MUST raise.

    A causal self-loop means "X causes itself" — physically meaningless.
    NOTEARS and the UI both choke on it.
    """
    with pytest.raises(ValidationError):
        CausalGraph(
            nodes=[_valid_node(id="a")],
            edges=[Edge(source="a", target="a", relation="CAUSES", confidence=50.0)],
        )


def test_causal_graph_valid_graph_passes():
    """Regression check: a well-formed graph must validate."""
    g = CausalGraph(
        nodes=[_valid_node(id="a"), _valid_node(id="b")],
        edges=[Edge(source="a", target="b", relation="CAUSES", confidence=50.0)],
    )
    assert len(g.nodes) == 2
    assert len(g.edges) == 1


def test_causal_graph_empty_passes():
    """Regression check: an empty graph (no nodes, no edges) must validate."""
    g = CausalGraph(nodes=[], edges=[])
    assert g.nodes == []
    assert g.edges == []


# ===========================================================================
# Edge.weight bounds + NaN/Inf rejection (SWARM-6 BUG #58).
# ===========================================================================

def test_edge_weight_has_bounds():
    """SWARM-6 BUG #58: `Edge.weight` MUST be bounded (e.g. [-1, 1]).

    A weight represents a correlation coefficient — it must be in [-1, 1].
    The current schema has no constraint and accepts 2.5, 1e9, -7, etc.
    """
    with pytest.raises(ValidationError):
        Edge(source="a", target="b", relation="CAUSES", confidence=50.0, weight=2.5)
    with pytest.raises(ValidationError):
        Edge(source="a", target="b", relation="CAUSES", confidence=50.0, weight=-2.5)


def test_edge_weight_accepts_in_range():
    """Regression check: weights within [-1, 1] validate."""
    for w in (-1.0, -0.5, 0.0, 0.5, 1.0):
        Edge(source="a", target="b", relation="CAUSES", confidence=50.0, weight=w)


def test_edge_weight_rejects_nan():
    """SWARM-6 BUG #58: `Edge(weight=float("nan"))` MUST raise.

    NaN propagates through every downstream Bayesian fit and silently
    poisons the heatmap. Pydantic does NOT reject NaN by default — the
    fix must add an explicit validator.
    """
    with pytest.raises(ValidationError):
        Edge(source="a", target="b", relation="CAUSES", confidence=50.0,
             weight=float("nan"))


def test_edge_weight_rejects_inf():
    """SWARM-6 BUG #58: `Edge(weight=float("inf"))` MUST raise."""
    with pytest.raises(ValidationError):
        Edge(source="a", target="b", relation="CAUSES", confidence=50.0,
             weight=float("inf"))
    with pytest.raises(ValidationError):
        Edge(source="a", target="b", relation="CAUSES", confidence=50.0,
             weight=float("-inf"))


# ===========================================================================
# Edge.relation normalization (SWARM-6 BUG #59).
# ===========================================================================

def test_edge_relation_normalized():
    """SWARM-6 BUG #59: `Edge(relation="causes")` MUST normalize to "CAUSES".

    The current schema is a bare `str` — case-insensitive comparisons
    need to be done everywhere downstream. Normalizing at the schema
    boundary is the cheapest fix.
    """
    e = Edge(source="a", target="b", relation="causes", confidence=50.0)
    assert e.relation == "CAUSES"


def test_edge_relation_trims_whitespace():
    """SWARM-6 BUG #59: leading/trailing whitespace MUST be stripped."""
    e = Edge(source="a", target="b", relation="  CAUSES  ", confidence=50.0)
    assert e.relation == "CAUSES"


def test_edge_relation_rejects_empty():
    """Empty relation string MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        Edge(source="a", target="b", relation="", confidence=50.0)


def test_edge_relation_rejects_whitespace_only():
    """Whitespace-only relation MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        Edge(source="a", target="b", relation="   ", confidence=50.0)


# ===========================================================================
# Node.confidence bounds (regression check).
# ===========================================================================

def test_node_confidence_in_valid_range():
    """Regression check: `Node(confidence=150)` MUST raise.

    The current schema already has `ge=0.0, le=100.0`, but this test
    locks the constraint in so it doesn't regress.
    """
    with pytest.raises(ValidationError):
        _valid_node(confidence=150.0)
    with pytest.raises(ValidationError):
        _valid_node(confidence=-1.0)


def test_node_confidence_accepts_in_range():
    """Regression check: confidence in [0, 100] validates."""
    for c in (0.0, 50.0, 100.0):
        _valid_node(confidence=c)


def test_node_id_required():
    """`Node(label="foo")` without `id` MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        Node(label="foo", confidence=50.0)


def test_node_label_required():
    """`Node(id="a")` without `label` MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        Node(id="a", confidence=50.0)