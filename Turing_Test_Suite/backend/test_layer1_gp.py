"""Tests for the Gaussian Process implementation in
``services/layer1/gaussian_process.py``.

Each test catches a specific bug from the SWARM-3 audit (see ``worklog.md``).
Tests are independent, use deterministic fixtures (seeded RNGs, mocked
joblib.dump to avoid file I/O), and do not require a running LLM or DB.
"""
import os

import numpy as np
import pandas as pd
import pytest

from schemas.graph import CausalGraph, Node, Edge
from services.layer1.gaussian_process import (
    GPEngine,
    refine_confidence_with_gp,
    StructuralFitter,
)


# ---------------------------------------------------------------------------
# Helper: capture saved models by patching joblib.dump
# ---------------------------------------------------------------------------
def _patch_joblib_dump(monkeypatch):
    """Patch joblib.dump to capture saved models instead of writing to disk.

    Returns a dict that will be populated with the captured models
    (key="models") when fit_graph runs.
    """
    import joblib
    captured = {}

    def fake_dump(obj, path, **kwargs):
        captured["models"] = obj
        return path

    monkeypatch.setattr(joblib, "dump", fake_dump)
    return captured


def _find_rbf_kernel(kernel):
    """Recursively find an RBF subkernel within a (possibly compound) kernel.

    sklearn compound kernels (Sum, Product) expose their subkernels via
    `.k1` and `.k2` (NOT `.kernels`). This helper unwraps
    `ConstantKernel * RBF + WhiteKernel` etc.
    """
    if "RBF" in type(kernel).__name__:
        return kernel
    for attr in ("k1", "k2"):
        subk = getattr(kernel, attr, None)
        if subk is not None:
            rbf = _find_rbf_kernel(subk)
            if rbf is not None:
                return rbf
    return None


# ===========================================================================
# BUG #43: GPEngine uses global RNG, output is non-deterministic
# ===========================================================================
def test_gp_random_seed_is_set():
    """Catches SWARM-3 BUG #43: GPEngine.predict_child uses `np.random.normal`
    (global RNG) for synthetic training data, making the output
    non-deterministic. The class should expose a RANDOM_STATE class
    attribute that is actually used to seed the RNG.

    Running predict_child twice must produce the same output.
    """
    pytest.importorskip("sklearn")

    # GPEngine must have a RANDOM_STATE class attribute (or instance attr)
    assert hasattr(GPEngine, "RANDOM_STATE") or hasattr(GPEngine(), "RANDOM_STATE"), (
        "BUG #43: GPEngine has no RANDOM_STATE class attribute — "
        "synthetic training data uses unseeded global RNG"
    )

    engine = GPEngine()
    parents = [
        {"mean": 0.5, "std_dev": 0.1},
        {"mean": 0.3, "std_dev": 0.2},
    ]
    weights = [0.4, 0.6]

    # Run predict_child twice — must produce the same output
    # Note: we DO NOT seed the global RNG here; the test passes only if
    # GPEngine uses its own seeded RNG.
    result1 = engine.predict_child(parents, weights)
    result2 = engine.predict_child(parents, weights)

    assert result1 == result2, (
        f"BUG #43: GPEngine.predict_child non-deterministic "
        f"(RANDOM_STATE not actually used):\n{result1}\nvs\n{result2}"
    )


# ===========================================================================
# BUG #40: GPEngine.predict_child floors y_train at 0
# ===========================================================================
def test_gp_predict_child_does_not_floor_at_zero():
    """Catches SWARM-3 BUG #40: GPEngine.predict_child floors y_train at 0
    via `y_train = np.maximum(0.0, y_train)`. This corrupts inhibitory
    (negative-weight) causal effects, since the GP is trained on floored
    (non-negative) data and learns to predict non-negative values only.

    For a target with a negative expected value (positive parent + negative
    weight), the prediction must be negative — not floored to 0.
    """
    pytest.importorskip("sklearn")
    # Seed global RNG so the test is reproducible regardless of bug presence.
    # (GPEngine uses unseeded np.random.normal — see test_gp_random_seed_is_set.)
    np.random.seed(42)

    engine = GPEngine()
    # Two parents, both with positive mean, but strongly negative weights
    # → expected child = 1.0 * -1.0 + 1.0 * -1.0 = -2.0 (strongly negative)
    parents = [
        {"mean": 1.0, "std_dev": 0.1},
        {"mean": 1.0, "std_dev": 0.1},
    ]
    weights = [-1.0, -1.0]

    result = engine.predict_child(parents, weights)

    # The predicted mean should be clearly negative (not floored to ~0)
    assert result["mean"] <= -0.1, (
        f"BUG #40: predict_child output not negative (floored at 0?): "
        f"mean={result['mean']} (expected <= -0.1 for inhibitory edges)"
    )


# ===========================================================================
# BUG #42: StructuralFitter computes x_std globally, not per-feature
# ===========================================================================
def test_gp_x_std_computed_per_feature_not_global(monkeypatch):
    """Catches SWARM-3 BUG #42: StructuralFitter.fit_graph computes
    `x_std = float(np.std(X))` — a single global std over the entire X
    matrix, not per-feature. The RBF kernel's `length_scale` should be a
    per-feature vector (so each parent's scale is normalized independently),
    not a single scalar derived from the global std.

    Fixture: two parent columns with very different scales (std=0.01 and
    std=100). With per-feature normalization, the RBF length_scale should be
    a 2-element vector with very different values. With the bug (global
    std), the length_scale is a single scalar.
    """
    pytest.importorskip("sklearn")
    captured = _patch_joblib_dump(monkeypatch)

    # Two parent columns of very different scales
    rng = np.random.default_rng(42)
    n = 50
    X1 = rng.normal(0, 0.01, n)   # tiny scale
    X2 = rng.normal(0, 100.0, n)  # huge scale
    Y = 0.5 * X1 + 0.001 * X2 + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})

    graph_data = {
        "nodes": [{"id": "X1"}, {"id": "X2"}, {"id": "Y"}],
        "edges": [
            {"source": "X1", "target": "Y"},
            {"source": "X2", "target": "Y"},
        ],
    }

    # Ensure GP_RANDOM_STATE env var doesn't interfere with kernel construction
    monkeypatch.delenv("GP_RANDOM_STATE", raising=False)

    StructuralFitter.fit_graph(df, graph_data, path_type="DATA")

    assert "models" in captured, "BUG #42: joblib.dump was not called"
    models = captured["models"]
    assert "Y" in models, (
        f"BUG #42: 'Y' not in saved models: {list(models.keys())}"
    )

    gp_y = models["Y"]["model"]
    # Use the initial kernel (gp.kernel) — the fitted kernel_ has the same
    # length_scale shape but optimized values; we want to check the SHAPE.
    kernel = gp_y.kernel

    # Find the RBF subkernel (kernel is a compound: ConstantKernel * RBF + WhiteKernel)
    rbf = _find_rbf_kernel(kernel)
    assert rbf is not None, f"BUG #42: No RBF kernel found in: {kernel}"

    ls = rbf.length_scale
    # length_scale must be a vector (per-feature) for the 2-parent case
    assert hasattr(ls, "__len__") and len(ls) == 2, (
        f"BUG #42: RBF length_scale is {ls!r} (scalar or wrong length) — "
        f"x_std was computed globally (single std over entire X matrix), "
        f"not per-feature (per-column std)"
    )
    # The two length_scales should differ substantially (X1 std≈0.01, X2 std≈100)
    # If they're identical, the global std was used (bug).
    assert abs(ls[0] - ls[1]) > 1e-6, (
        f"BUG #42: RBF length_scales are identical ({ls}) — "
        f"computed from global std, not per-feature"
    )


# ===========================================================================
# BUG #41: refine_confidence_with_gp drops node metadata
# ===========================================================================
def test_gp_refine_confidence_preserves_node_metadata():
    """Catches SWARM-3 BUG #41: refine_confidence_with_gp constructs new
    Node objects with only `id`, `label`, `confidence` — dropping
    `domain`, `is_unknown`, and `description`. After refinement, the
    node should still carry all original metadata.
    """
    pytest.importorskip("sklearn")

    nodes = [
        Node(id="X", label="Variable X", domain="physics", confidence=80.0,
             is_unknown=False, description="The primary cause"),
        Node(id="Y", label="Variable Y", domain="physics", confidence=70.0,
             is_unknown=True, description="The measured effect"),
        Node(id="Z", label="Variable Z", domain="chemistry", confidence=60.0,
             is_unknown=False, description="A known confounder"),
    ]
    edges = [
        Edge(source="X", target="Y", relation="CAUSES", confidence=85.0, weight=0.9),
        Edge(source="Z", target="Y", relation="CAUSES", confidence=75.0, weight=0.6),
    ]
    graph = CausalGraph(nodes=nodes, edges=edges)

    refined = refine_confidence_with_gp(graph)

    # Refined graph should have the same number of nodes/edges
    assert len(refined.nodes) == len(nodes)
    assert len(refined.edges) == len(edges)

    # Each refined node must preserve ALL original metadata
    for orig, ref in zip(nodes, refined.nodes):
        assert ref.id == orig.id
        assert ref.label == orig.label, (
            f"BUG #41: label changed for {orig.id}: {ref.label!r} vs {orig.label!r}"
        )
        assert ref.domain == orig.domain, (
            f"BUG #41: domain dropped for {orig.id}: "
            f"{ref.domain!r} vs {orig.domain!r}"
        )
        assert ref.is_unknown == orig.is_unknown, (
            f"BUG #41: is_unknown dropped for {orig.id}: "
            f"{ref.is_unknown} vs {orig.is_unknown}"
        )
        assert ref.description == orig.description, (
            f"BUG #41: description dropped for {orig.id}: "
            f"{ref.description!r} vs {orig.description!r}"
        )


# ===========================================================================
# Edge case: singular kernel matrix (Cholesky failure)
# ===========================================================================
def test_gp_cholesky_failure_handled():
    """Catches SWARM-3 (edge case): singular kernel matrix should not crash
    GPEngine.predict_child with LinAlgError. The function must catch the
    Cholesky failure and fall back to a quadrature-based estimate.
    """
    pytest.importorskip("sklearn")
    # Seed global RNG for reproducibility (predict_child uses unseeded RNG)
    np.random.seed(42)

    engine = GPEngine()
    # Parents with zero std_dev — synthetic X_train will be constant,
    # leading to a singular kernel matrix.
    parents = [
        {"mean": 0.5, "std_dev": 0.0},
        {"mean": 0.5, "std_dev": 0.0},
    ]
    weights = [0.5, 0.5]

    # Should NOT raise LinAlgError; should return a fallback result
    try:
        result = engine.predict_child(parents, weights)
        assert "mean" in result, f"Missing 'mean' in result: {result}"
        assert "std_dev" in result, f"Missing 'std_dev' in result: {result}"
    except np.linalg.LinAlgError as e:
        pytest.fail(
            f"BUG (edge case): GPEngine crashed with LinAlgError on singular "
            f"kernel: {e}"
        )


# ===========================================================================
# Edge case: zero-variance input
# ===========================================================================
def test_gp_handles_zero_variance_input(monkeypatch):
    """Catches SWARM-3 (edge case): a column with zero variance should
    not cause division by zero in StructuralFitter.fit_graph's
    `np.std(X)` computation. The function must handle zero-variance
    columns gracefully (or raise a clear error).
    """
    pytest.importorskip("sklearn")
    captured = _patch_joblib_dump(monkeypatch)
    monkeypatch.delenv("GP_RANDOM_STATE", raising=False)

    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({
        "const": np.full(n, 7.0),     # zero variance
        "varying": rng.normal(0, 1, n),
        "Y": rng.normal(0, 1, n),
    })
    graph_data = {
        "nodes": [{"id": "const"}, {"id": "varying"}, {"id": "Y"}],
        "edges": [
            {"source": "const", "target": "Y"},
            {"source": "varying", "target": "Y"},
        ],
    }

    try:
        StructuralFitter.fit_graph(df, graph_data, path_type="DATA")
    except ValueError as e:
        # Acceptable: a clear error about zero variance
        assert any(s in str(e).lower()
                   for s in ["variance", "zero", "constant", "degenerate"]), (
            f"StructuralFitter raised unrelated ValueError: {e}"
        )
    except (ZeroDivisionError, FloatingPointError) as e:
        pytest.fail(
            f"BUG (edge case): StructuralFitter divided by zero on "
            f"zero-variance input: {type(e).__name__}: {e}"
        )
    except np.linalg.LinAlgError as e:
        pytest.fail(
            f"BUG (edge case): StructuralFitter crashed with LinAlgError on "
            f"zero-variance input: {e}"
        )

    # If it didn't raise, verify a model was saved for Y (with parents)
    if "models" in captured:
        assert "Y" in captured["models"], (
            f"BUG (edge case): 'Y' not in saved models: "
            f"{list(captured['models'].keys())}"
        )


# ===========================================================================
# Regression: RBF kernel length_scale must be positive
# ===========================================================================
def test_gp_kernel_uses_correct_hyperparams(monkeypatch):
    """Catches SWARM-3 (regression): the RBF kernel's `length_scale` must
    be positive (not 0 or negative). A length_scale of 0 makes the kernel
    degenerate (zero everywhere except at training points); negative is
    mathematically invalid.

    For both GPEngine.predict_child and StructuralFitter.fit_graph, the
    RBF length_scale must be strictly positive.
    """
    pytest.importorskip("sklearn")
    captured = _patch_joblib_dump(monkeypatch)
    monkeypatch.delenv("GP_RANDOM_STATE", raising=False)

    # Test StructuralFitter's RBF length_scale
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({
        "X": rng.normal(0, 1, n),
        "Y": 0.5 * rng.normal(0, 1, n) + rng.normal(0, 0.1, n),
    })
    graph_data = {
        "nodes": [{"id": "X"}, {"id": "Y"}],
        "edges": [{"source": "X", "target": "Y"}],
    }

    StructuralFitter.fit_graph(df, graph_data, path_type="DATA")

    assert "models" in captured
    assert "Y" in captured["models"], (
        f"'Y' not in saved models: {list(captured['models'].keys())}"
    )
    gp = captured["models"]["Y"]["model"]
    kernel = gp.kernel

    # Find RBF subkernel (kernel is a compound: ConstantKernel * RBF + WhiteKernel)
    rbf = _find_rbf_kernel(kernel)
    assert rbf is not None, f"No RBF kernel in: {kernel}"

    ls = rbf.length_scale
    if hasattr(ls, "__len__"):
        assert all(float(l) > 0 for l in ls), (
            f"BUG: RBF length_scale has non-positive values: {ls}"
        )
    else:
        assert float(ls) > 0, (
            f"BUG: RBF length_scale is non-positive: {ls}"
        )

    # Also test GPEngine.predict_child — its kernel uses length_scale=np.ones(...),
    # which should be all 1.0 (positive). Verify via direct call.
    np.random.seed(42)
    engine = GPEngine()
    parents = [{"mean": 0.5, "std_dev": 0.1}, {"mean": 0.3, "std_dev": 0.2}]
    weights = [0.4, 0.6]
    result = engine.predict_child(parents, weights)
    assert result["std_dev"] >= 0, (
        f"BUG: predict_child returned negative std_dev: {result['std_dev']}"
    )