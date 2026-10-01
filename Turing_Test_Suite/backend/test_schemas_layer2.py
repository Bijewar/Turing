"""
Pydantic schema tests for `schemas/layer2.py`.

Catches:
- SWARM-6 BUG #60 — `Layer2Request.max_iterations` has no upper bound.
- SWARM-6 BUG #61 — `HeatmapInput.resolution` has no upper bound.
- SWARM-6 BUG (related) — `AgentEntry` schema missing or has unconstrained
  `type` field. Layer 2 UI is fed agents with `type` strings; an
  explicit schema (with enum) protects the frontend renderer.
- SWARM-6 BUG (related) — `OptimizerState` (a.k.a. BayesianOptimizerState
  in the frontend) is referenced by the UI but has no schema; the
  backend serializes a dict that the frontend reads as an object.
"""
import pytest
from pydantic import ValidationError


from schemas.graph import CausalGraph
from schemas.layer2 import (
    HeatmapInput,
    Layer2Request,
    SearchSpace,
)


def _valid_graph() -> CausalGraph:
    """Helper — returns a small valid CausalGraph."""
    from schemas.graph import Node, Edge
    return CausalGraph(
        nodes=[Node(id="a", label="A", confidence=50.0)],
        edges=[],
    )


# ===========================================================================
# Layer2Request.max_iterations bounds (SWARM-6 BUG #60).
# ===========================================================================

def test_layer2_request_max_iterations_bounded():
    """SWARM-6 BUG #60: `max_iterations=1000000` MUST raise ValidationError.

    The Layer 2 orchestrator runs one Bayesian round per iteration and
    every iteration drives ≥3 LLM calls. 1,000,000 iterations = >3 million
    LLM calls = thousands of dollars per request. The fix bounds the
    field to (1, 50).
    """
    with pytest.raises(ValidationError):
        Layer2Request(graph=_valid_graph(), target_node_id="a",
                      max_iterations=1_000_000)


def test_layer2_request_max_iterations_upper_boundary():
    """Upper boundary check: `max_iterations=50` MUST validate."""
    obj = Layer2Request(graph=_valid_graph(), target_node_id="a", max_iterations=50)
    assert obj.max_iterations == 50


def test_layer2_request_max_iterations_above_boundary_rejected():
    """`max_iterations=51` MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        Layer2Request(graph=_valid_graph(), target_node_id="a", max_iterations=51)


def test_layer2_request_max_iterations_ge_1():
    """`max_iterations=0` MUST raise ValidationError (lower bound)."""
    with pytest.raises(ValidationError):
        Layer2Request(graph=_valid_graph(), target_node_id="a", max_iterations=0)


def test_layer2_request_max_iterations_negative_rejected():
    """Negative values MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        Layer2Request(graph=_valid_graph(), target_node_id="a", max_iterations=-5)


def test_layer2_request_max_iterations_default_10():
    """Regression check: default is 10 when omitted."""
    obj = Layer2Request(graph=_valid_graph(), target_node_id="a")
    assert obj.max_iterations == 10


# ===========================================================================
# HeatmapInput.resolution bounds (SWARM-6 BUG #61).
# ===========================================================================

def test_heatmap_resolution_bounded():
    """SWARM-6 BUG #61: `resolution=10000` MUST raise ValidationError.

    `resolution` is the number of points per axis. The heatmap allocates
    `resolution²` HeatmapPoint objects; 10000² = 100M points = OOM risk.
    The fix caps it (e.g. at 200).
    """
    with pytest.raises(ValidationError):
        HeatmapInput(
            nodes=["a", "b"],
            edges=[],
            domain_config={"a": SearchSpace(min=0, max=10)},
            resolution=10_000,
        )


def test_heatmap_resolution_upper_boundary():
    """Upper boundary check: `resolution=200` MUST validate."""
    obj = HeatmapInput(
        nodes=["a", "b"],
        edges=[],
        domain_config={"a": SearchSpace(min=0, max=10)},
        resolution=200,
    )
    assert obj.resolution == 200


def test_heatmap_resolution_above_boundary_rejected():
    """`resolution=201` MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        HeatmapInput(
            nodes=["a", "b"],
            edges=[],
            domain_config={"a": SearchSpace(min=0, max=10)},
            resolution=201,
        )


def test_heatmap_resolution_min_2():
    """Regression check: `resolution=1` MUST raise ValidationError (already enforced)."""
    with pytest.raises(ValidationError):
        HeatmapInput(
            nodes=["a", "b"],
            edges=[],
            domain_config={"a": SearchSpace(min=0, max=10)},
            resolution=1,
        )


def test_heatmap_resolution_default_10():
    """Regression check: default resolution is 10."""
    obj = HeatmapInput(
        nodes=["a", "b"],
        edges=[],
        domain_config={"a": SearchSpace(min=0, max=10)},
    )
    assert obj.resolution == 10


# ===========================================================================
# AgentEntry — must exist with a closed-enum `type` field.
# ===========================================================================

def test_agent_entry_type_valid_values():
    """SWARM-6 BUG: `AgentEntry(type="invalid")` MUST raise ValidationError.

    The Layer 2 UI renders agents by `type` (explorer / exploiter /
    contrarian / skeptic). The backend currently serializes a free-form
    dict; if a typo or attacker-supplied string slips through, the UI
    crashes (variant == undefined → render error). The fix introduces a
    proper `AgentEntry` Pydantic model with a closed `type` enum.
    """
    from schemas.layer2 import AgentEntry  # noqa: F401 — import must succeed
    from schemas.layer2 import AgentEntry as AE
    with pytest.raises(ValidationError):
        AE(id="x", type="invalid", agentId="A1", y=10, content="hi", delayMs=0)


def test_agent_entry_type_includes_explorer_exploiter_contrarian():
    """Valid `type` values MUST include explorer, exploiter, contrarian."""
    from schemas.layer2 import AgentEntry as AE
    AE(id="x", type="explorer", agentId="A1", y=10, content="hi", delayMs=0)
    AE(id="x", type="exploiter", agentId="A1", y=10, content="hi", delayMs=0)
    AE(id="x", type="contrarian", agentId="A1", y=10, content="hi", delayMs=0)


def test_agent_entry_type_includes_skeptic_if_used():
    """If the codebase routes `skeptic` to the UI variant map, the enum
    must include `skeptic` as a valid value (otherwise it would be a
    silent failure).

    SWARM-6 BUG: `routers/runs.py:263` checks `role == 'contrarian' or
    role == 'skeptic'` — the backend CAN emit `skeptic` as a `type`,
    but the frontend's TypeScript `AgentEntry.type` union is
    `'explorer' | 'exploiter' | 'contrarian'` (no skeptic). The Pydantic
    schema fix should include skeptic to match the backend's behaviour.
    """
    from schemas.layer2 import AgentEntry as AE

    # Source-level check: confirm `skeptic` is used somewhere in the
    # Layer 2 UI code path.
    import inspect
    import routers.runs as runs_mod
    src = inspect.getsource(runs_mod)
    skeptic_used = "skeptic" in src

    if skeptic_used:
        # If backend emits `skeptic`, the schema must accept it.
        AE(id="x", type="skeptic", agentId="A1", y=10, content="hi", delayMs=0)
    # If skeptic is NOT used anywhere, this test is a no-op.


# ===========================================================================
# Layer2Request required fields.
# ===========================================================================

def test_layer2_request_target_node_id_required():
    """`Layer2Request(target_node_id=None)` MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        Layer2Request(graph=_valid_graph(), target_node_id=None)


def test_layer2_request_target_node_id_empty_string_rejected():
    """Empty target_node_id MUST raise ValidationError (currently allowed)."""
    with pytest.raises(ValidationError):
        Layer2Request(graph=_valid_graph(), target_node_id="")


def test_layer2_request_graph_required():
    """`Layer2Request(graph=None)` MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        Layer2Request(graph=None, target_node_id="a")


def test_layer2_request_graph_validates_internal_structure():
    """SWARM-6 BUG #57 (cascading): the graph attribute should validate
    internal structure (no duplicate node IDs, etc.) — verified by
    constructing a Layer2Request with a malformed graph."""
    from schemas.graph import Node
    # Duplicate node IDs — must propagate up via nested validation.
    with pytest.raises(ValidationError):
        Layer2Request(
            graph=CausalGraph(
                nodes=[Node(id="a", label="A", confidence=50.0),
                       Node(id="a", label="A", confidence=50.0)],
                edges=[],
            ),
            target_node_id="a",
        )


# ===========================================================================
# OptimizerState — must exist with required fields validated.
# ===========================================================================

def test_optimizer_state_required_fields_validated():
    """The frontend `BayesianOptimizerState` interface defines
    `targetMaximizationPercentage: number`. The backend currently
    returns a free-form dict; the fix introduces a Pydantic schema
    `OptimizerState` with that field.

    Construction with `targetMaximizationPercentage=None` MUST raise.
    """
    from schemas.layer2 import OptimizerState  # noqa: F401
    from schemas.layer2 import OptimizerState as OS
    with pytest.raises(ValidationError):
        OS(targetMaximizationPercentage=None)


def test_optimizer_state_target_maximization_bounded():
    """`targetMaximizationPercentage` must be in [0, 100]."""
    from schemas.layer2 import OptimizerState as OS
    with pytest.raises(ValidationError):
        OS(targetMaximizationPercentage=-1.0)
    with pytest.raises(ValidationError):
        OS(targetMaximizationPercentage=150.0)


# ===========================================================================
# SearchSpace — regression checks (already validates).
# ===========================================================================

def test_search_space_min_lt_max():
    """Regression check: `SearchSpace(min=5, max=1)` MUST raise."""
    with pytest.raises(ValidationError):
        SearchSpace(min=5, max=1)


def test_search_space_min_eq_max_rejected():
    """Regression check: `SearchSpace(min=5, max=5)` MUST raise."""
    with pytest.raises(ValidationError):
        SearchSpace(min=5, max=5)