"""
test_layer2_do_calculus.py

Tests for `python-engine/services/layer2/do_calculus.py`.

Each test names the exact SWARM-4 BUG ID it targets (see Turing_Bug_Audit_Report.md,
Task ID: SWARM-4). Tests are deterministic (seeds set in the autouse fixture)
and do not require LLM API keys.
"""
import os
import sys
import warnings
import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Module path bootstrap — module-local, does not touch conftest.py
# ---------------------------------------------------------------------------
_PY_ENGINE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "upload", "Turing-main", "python-engine")
)
if _PY_ENGINE_DIR not in sys.path:
    sys.path.insert(0, _PY_ENGINE_DIR)

from schemas.layer2 import GraphEdge, GaussianPrediction  # noqa: E402
from services.layer2.do_calculus import DoCalculusSimulator  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _deterministic_rng(monkeypatch):
    """Seed Python's `random`, NumPy's global RNG, and the GP env var so the
    GPEngine synthetic training data is reproducible (BUG #11).
    """
    monkeypatch.setenv("GP_RANDOM_STATE", "42")
    np.random.seed(42)
    import random as _random
    _random.seed(42)
    # Suppress the cascade of UserWarnings the buggy code emits, so test
    # output is readable.  Tests that ASSERT on warnings re-enable them.
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    np.random.seed(None)
    warnings.resetwarnings()


def _edge(source, target, weight=1.0):
    """Helper: build a GraphEdge (defaults to weight=1.0 — see BUG #10)."""
    return GraphEdge(source=source, target=target, weight=weight)


# ---------------------------------------------------------------------------
# 1. SWARM-4 BUG #1: non-intervened source nodes silently dropped
# ---------------------------------------------------------------------------
def test_does_not_drop_non_intervened_source_nodes():
    """SWARM-4 BUG #1 (Critical).

    The buggy forward-propagation loop skips setting `predictions[node]`
    whenever `parent_preds == []`.  For a SOURCE node (no incoming edges),
    `parent_preds` is ALWAYS empty — so non-intervened source nodes are
    silently dropped, and any descendant whose only parent is that source
    is also dropped.

    The fixed code MUST assign a prediction (default value + base_noise
    std_dev) to every non-intervened source node so the predictions dict
    covers the full graph as the schema docstring promises.

    This test builds A→B→C with intervention only on B and asserts:
      - 'A' is present in predictions (not dropped)
      - 'C' is present in predictions (not dropped because B is predicted)
      - 'B' has the intervened value
    An LLM that "fixes" the bug by injecting 0.0 for ALL missing nodes
    (including the non-source 'C') will fail the second assertion's
    assertion that 'B' actually equals the intervention value.
    """
    sim = DoCalculusSimulator()
    nodes = ["A", "B", "C"]
    edges = [_edge("A", "B"), _edge("B", "C")]
    res = sim.simulate(nodes=nodes, edges=edges, interventions={"B": 1.0})

    preds = res.predictions
    # Non-intervened source 'A' must NOT be dropped
    assert "A" in preds, (
        "BUG #1: Non-intervened source node 'A' was silently dropped from "
        "predictions.  Source nodes need a default value (e.g. domain center "
        "or 0.0) — they must not be silently omitted."
    )
    # Intervened node 'B' must equal the intervention value exactly
    assert "B" in preds
    assert preds["B"].mean == pytest.approx(1.0, abs=1e-9), (
        f"Intervened node 'B' should be exactly 1.0, got {preds['B'].mean}"
    )
    assert preds["B"].std_dev == pytest.approx(0.0, abs=1e-12), (
        f"Intervened node 'B' should have std_dev=0 (deterministic), got {preds['B'].std_dev}"
    )
    # Descendant 'C' must be present (B is predicted, so C should propagate)
    assert "C" in preds, (
        "BUG #1 cascade: descendant 'C' was dropped because its parent 'B' "
        "is predicted but the propagation skipped C — check the topo loop."
    )


# ---------------------------------------------------------------------------
# 2. Backdoor adjustment — non-intervened confounder must NOT be dropped
# ---------------------------------------------------------------------------
def test_backdoor_adjustment_correct():
    """SWARM-4 BUG #1 (Critical) — backdoor adjustment scenario.

    Graph:  Z → X,  X → Y,  Z → Y   (Z is a confounder of X and Y)

    Intervene on X = 5.  The CAUSAL (backdoor-adjusted) effect of X on Y is
    the direct effect X*w_XY plus the contribution of Z (the confounder),
    using whatever default the fix assigns to non-intervened sources.

    The buggy code DROPS Z entirely (BUG #1), so:
        pred_Y (buggy)  = X * w_XY  (missing Z's contribution)

    The fix must INCLUDE Z in predictions.  We verify:
      (a) 'Z' is present in the predictions dict (catches BUG #1).
      (b) When Z is also intervened to a non-zero value, pred_Y changes —
          i.e. the simulator correctly propagates Z's effect (positive case).
      (c) pred_Y(do(X=5)) is close to 5*w_XY + Z_default*w_ZY — for ANY
          default the fix chooses, pred_Y must depend on Z.  We verify the
          dependency by comparing two scenarios.

    A fake fix that hardcodes 0.0 for Z and drops it from the dict will fail
    assertion (a).  A fake fix that injects 0.0 but keeps Z in the dict will
    fail assertion (b) (pred_Y won't change when Z is intervened differently).
    """
    sim = DoCalculusSimulator()
    nodes = ["X", "Y", "Z"]
    edges = [
        _edge("Z", "X", weight=1.0),
        _edge("X", "Y", weight=1.0),
        _edge("Z", "Y", weight=10.0),  # large confounding effect
    ]

    # Scenario 1: intervene only on X
    res1 = sim.simulate(nodes=nodes, edges=edges, interventions={"X": 5.0})
    # (a) Z must be in predictions (BUG #1 fix)
    assert "Z" in res1.predictions, (
        "BUG #1: Confounder Z (non-intervened source) was dropped — the "
        "backdoor adjustment is wrong because Z's contribution to Y is missing."
    )
    pred_Y1 = res1.predictions["Y"].mean
    assert pred_Y1 is not None, "Y prediction missing"

    # (b) Intervening Z too must change Y's prediction (Z is propagated)
    np.random.seed(42)
    res2 = sim.simulate(
        nodes=nodes, edges=edges, interventions={"X": 5.0, "Z": 10.0}
    )
    pred_Y2 = res2.predictions["Y"].mean
    assert abs(pred_Y2 - pred_Y1) > 1.0, (
        f"Backdoor adjustment broken: pred_Y did not change when Z was "
        f"intervened (pred_Y1={pred_Y1}, pred_Y2={pred_Y2}).  Z's contribution "
        "to Y is missing — the simulator is not propagating Z."
    )

    # (c) The causal effect of X (the direct effect) must be captured:
    #     pred_Y(do(X=10)) - pred_Y(do(X=0)) should ≈ 10 * w_XY = 10
    np.random.seed(42)
    res_x0 = sim.simulate(nodes=nodes, edges=edges, interventions={"X": 0.0})
    np.random.seed(42)
    res_x10 = sim.simulate(nodes=nodes, edges=edges, interventions={"X": 10.0})
    delta = res_x10.predictions["Y"].mean - res_x0.predictions["Y"].mean
    assert delta == pytest.approx(10.0, abs=2.0), (
        f"Direct causal effect of X on Y should be ~10 (10 * w_XY=1.0), "
        f"got {delta}.  The do-calculus must propagate the intervention."
    )


# ---------------------------------------------------------------------------
# 3. Front-door adjustment — unobserved confounder must NOT pollute Y
# ---------------------------------------------------------------------------
def test_front_door_adjustment_correct():
    """SWARM-4 BUG #1 (Critical) — front-door adjustment scenario.

    Graph:  X → M → Y,  U → Y   (U is an unobserved confounder)

    Intervene on X = 5.  The front-door causal effect of X on Y is the
    indirect effect through M:
        pred_Y(do(X)) = X * w_XM * w_MY   (+ U_default * w_UY, but U is
                                            unobserved so its default MUST
                                            be 0.0 or it pollutes the result)

    The buggy code DROPS U (BUG #1), which gives the right answer by
    accident.  A naive fix that defaults ALL non-intervened sources to a
    non-zero value (e.g. 1.0) will pollute Y and give the WRONG answer.

    This test sets w_UY very large (50.0) so any non-zero U default would
    be obvious.  It asserts:
      (a) U is in the predictions dict (catches BUG #1).
      (b) pred_Y ≈ 5 * w_XM * w_MY = 5 (U's contribution is zero, not 5*50=250).

    A fake fix that defaults U to 1.0 will fail (b): pred_Y ≈ 5 + 50 = 55.
    """
    sim = DoCalculusSimulator()
    nodes = ["X", "M", "Y", "U"]
    edges = [
        _edge("X", "M", weight=1.0),
        _edge("M", "Y", weight=1.0),
        _edge("U", "Y", weight=50.0),  # HUGE confounding effect
    ]
    res = sim.simulate(nodes=nodes, edges=edges, interventions={"X": 5.0})

    # (a) U must be in predictions (BUG #1 fix) — even though it is unobserved,
    #     the schema promises "every node in the graph".
    assert "U" in res.predictions, (
        "BUG #1: Unobserved confounder U (non-intervened source) was dropped "
        "from predictions."
    )
    # (b) Y must equal the front-door (indirect) effect, NOT polluted by U.
    pred_Y = res.predictions["Y"].mean
    assert pred_Y == pytest.approx(5.0, abs=1.0), (
        f"Front-door effect broken: pred_Y should be ~5 (X * w_XM * w_MY = "
        f"5 * 1 * 1), got {pred_Y}.  The unobserved confounder U is leaking "
        "into Y — U's default value must be 0.0 (or U must not be propagated)."
    )


# ---------------------------------------------------------------------------
# 4. SWARM-4 BUG #10 (sibling): intervention value is actually applied
# ---------------------------------------------------------------------------
def test_intervention_value_applied():
    """SWARM-4 BUG #10 sibling (High) — `do(X=5)` must set X to 5, not the
    observational mean or any other value.

    The buggy code already does this correctly for the intervened node
    itself (predictions[X] = GaussianPrediction(mean=5, std_dev=0)).  This
    test is a contract test that the bugfix preserves the behavior.  An LLM
    that "fixes" BUG #1 by overriding intervened nodes with defaults will
    fail this test.
    """
    sim = DoCalculusSimulator()
    nodes = ["A", "B"]
    edges = [_edge("A", "B", weight=2.0)]
    res = sim.simulate(nodes=nodes, edges=edges, interventions={"A": 7.5})

    assert "A" in res.predictions
    assert res.predictions["A"].mean == pytest.approx(7.5, abs=1e-12), (
        f"Intervention do(A=7.5) not applied: predictions[A].mean = "
        f"{res.predictions['A'].mean} instead of 7.5"
    )
    assert res.predictions["A"].std_dev == pytest.approx(0.0, abs=1e-12), (
        f"Intervened node should have std_dev=0 (deterministic), got "
        f"{res.predictions['A'].std_dev}"
    )


# ---------------------------------------------------------------------------
# 5. No interventions returns observational
# ---------------------------------------------------------------------------
def test_no_intervention_returns_observational():
    """Contract test — empty interventions dict must not crash and must
    return the observational (default-source-values) prediction.

    With the buggy code, ALL source nodes are dropped (BUG #1) and the
    predictions dict is empty (or near-empty).  The fix must populate every
    node with its default/observational value.
    """
    sim = DoCalculusSimulator()
    nodes = ["A", "B"]
    edges = [_edge("A", "B", weight=1.0)]
    # No interventions — purely observational
    res = sim.simulate(nodes=nodes, edges=edges, interventions={})

    # Both nodes must be present (the fix populates sources with defaults)
    assert "A" in res.predictions, (
        "BUG #1: With no interventions, source node 'A' was dropped — the "
        "observational run must populate every source with a default."
    )
    assert "B" in res.predictions, (
        "BUG #1 cascade: 'B' was dropped because 'A' was dropped."
    )


# ---------------------------------------------------------------------------
# 6. Predictions for all descendants
# ---------------------------------------------------------------------------
def test_predictions_for_all_descendants():
    """Contract test — every descendant of an intervened node gets a fresh
    prediction (the intervention propagates through the chain).
    """
    sim = DoCalculusSimulator()
    # Linear chain A → B → C → D
    nodes = ["A", "B", "C", "D"]
    edges = [_edge("A", "B"), _edge("B", "C"), _edge("C", "D")]
    res = sim.simulate(nodes=nodes, edges=edges, interventions={"A": 3.0})

    # All descendants of A (B, C, D) must have predictions
    for n in ["B", "C", "D"]:
        assert n in res.predictions, (
            f"Descendant '{n}' of intervened 'A' is missing from predictions."
        )


# ---------------------------------------------------------------------------
# 7. Predictions for non-descendants unchanged
# ---------------------------------------------------------------------------
def test_predictions_for_non_descendants_unchanged():
    """SWARM-4 BUG #1 (Critical) — non-descendants of the intervened node
    must NOT be affected by the intervention, but they MUST still be present
    in the predictions dict (with their own default values).

    The buggy code drops ALL non-intervened sources — so the disconnected
    subgraph is missing entirely.  The fix must populate it with defaults
    that do NOT depend on the intervention.

    This test builds two disconnected subgraphs A→B and C→D, intervenes on
    A, and verifies:
      - C and D are present (BUG #1 fix — non-descendants not dropped).
      - D's value does NOT change when A's intervention changes.
    """
    sim = DoCalculusSimulator()
    nodes = ["A", "B", "C", "D"]
    edges = [_edge("A", "B"), _edge("C", "D")]

    # Intervene on A=1.0
    np.random.seed(42)
    res1 = sim.simulate(nodes=nodes, edges=edges, interventions={"A": 1.0})
    # Intervene on A=99.0 (very different)
    np.random.seed(42)
    res2 = sim.simulate(nodes=nodes, edges=edges, interventions={"A": 99.0})

    # C and D must both be present (BUG #1 fix)
    assert "C" in res1.predictions, (
        "BUG #1: Non-descendant source 'C' was dropped from predictions."
    )
    assert "D" in res1.predictions, (
        "BUG #1 cascade: Non-descendant 'D' was dropped (its parent 'C' "
        "was dropped)."
    )
    # D's value should NOT change with A's intervention
    d1 = res1.predictions["D"].mean
    d2 = res2.predictions["D"].mean
    assert d1 == pytest.approx(d2, abs=0.1), (
        f"Non-descendant 'D' changed from {d1} to {d2} when A was "
        f"intervened differently — non-descendants must be independent of A."
    )
    # B (descendant of A) MUST change
    b1 = res1.predictions["B"].mean
    b2 = res2.predictions["B"].mean
    assert abs(b2 - b1) > 50.0, (
        f"Descendant 'B' did not change ({b1} → {b2}) when A was intervened "
        f"differently — propagation is broken."
    )


# ---------------------------------------------------------------------------
# 8. Graph with no interventions dict (or empty) doesn't crash
# ---------------------------------------------------------------------------
def test_handles_graph_with_no_interventions():
    """SWARM-4 BUG #1 — graph with no interventions must not crash and must
    return predictions for every node (using default source values).
    """
    sim = DoCalculusSimulator()
    nodes = ["A", "B", "C"]
    edges = [_edge("A", "B"), _edge("B", "C")]
    # Empty interventions dict — the simulator must accept this and not crash
    res = sim.simulate(nodes=nodes, edges=edges, interventions={})

    # All nodes must be present
    for n in nodes:
        assert n in res.predictions, (
            f"BUG #1: With no interventions, node '{n}' is missing — the "
            "observational run must populate every node."
        )


# ---------------------------------------------------------------------------
# 9. Cycle in graph raises clear error
# ---------------------------------------------------------------------------
def test_handles_cycle_in_graph_raises_error():
    """Contract test — do-calculus requires a DAG.  A graph with a cycle that
    SURVIVES the intervention (i.e. a cycle among non-intervened nodes) must
    raise a clear ValueError (not silently loop, not crash with a KeyError).

    Note: an intervention CUTS incoming edges to the intervened node, so a
    cycle that goes through an intervened node is broken by the
    intervention and is NOT a real cycle in the active graph.  We use a
    cycle among NON-intervened nodes (B ↔ C) plus a single source A→B with
    A intervened, so the cycle is preserved in the active subgraph.
    """
    sim = DoCalculusSimulator()
    # A → B; B ↔ C (cycle)
    nodes = ["A", "B", "C"]
    edges = [
        _edge("A", "B"),
        _edge("B", "C"),
        _edge("C", "B"),  # back-edge forming cycle B↔C
    ]
    with pytest.raises(ValueError, match="cycle"):
        sim.simulate(nodes=nodes, edges=edges, interventions={"A": 1.0})


# ---------------------------------------------------------------------------
# 10. Simulation output has correct shape
# ---------------------------------------------------------------------------
def test_simulation_results_have_correct_shape():
    """SWARM-4 BUG #14 — SimulationStepOutput must have a `predictions` dict.

    The schema docstring promises "Mean and std_dev for every node in the
    graph".  After BUG #1 is fixed, this becomes literally true.  This test
    verifies the output shape AND catches BUG #14 by using a graph where
    the intervened node is NOT a source — so a non-intervened source 'A'
    exists and must NOT be dropped.

    Setup: A → B → C, intervene on B (B is intermediate, A is the
    non-intervened source).  The buggy code drops A (BUG #1) and the
    shape assertion fails.  The fix must populate A with a default.
    """
    sim = DoCalculusSimulator()
    nodes = ["A", "B", "C"]
    edges = [_edge("A", "B"), _edge("B", "C")]
    # Intervene on the INTERMEDIATE node B (A is a non-intervened source)
    res = sim.simulate(nodes=nodes, edges=edges, interventions={"B": 1.0})

    # Output must have `predictions` dict
    assert hasattr(res, "predictions"), "SimulationStepOutput missing 'predictions'"
    assert isinstance(res.predictions, dict), "predictions must be a dict"

    # Each value is a GaussianPrediction
    for k, v in res.predictions.items():
        assert isinstance(v, GaussianPrediction), (
            f"predictions['{k}'] is {type(v)}, expected GaussianPrediction"
        )
        assert hasattr(v, "mean")
        assert hasattr(v, "std_dev")
        assert isinstance(v.mean, (int, float))
        assert isinstance(v.std_dev, (int, float))
        assert v.std_dev >= 0.0, f"std_dev must be non-negative, got {v.std_dev}"

    # Intervened node 'B' must be present with the exact intervention value
    assert "B" in res.predictions
    assert res.predictions["B"].mean == pytest.approx(1.0, abs=1e-12)
    assert res.predictions["B"].std_dev == pytest.approx(0.0, abs=1e-12)

    # BUG #14: After the fix, EVERY node in the graph should be present
    # (the schema docstring promises this).  The buggy code drops
    # non-intervened sources — this assertion catches that.  We chose a
    # graph with a non-intervened source 'A' specifically to trigger this.
    for n in nodes:
        assert n in res.predictions, (
            f"BUG #14: predictions is missing node '{n}'.  The schema "
            "docstring promises 'Mean and std_dev for every node in the "
            "graph' — either populate every node or update the schema."
        )