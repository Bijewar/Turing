"""Tests for the NOTEARS implementation in
``services/layer1/notears.py``.

Each test catches a specific bug from the SWARM-3 audit (see ``worklog.md``).
Tests are independent, use deterministic fixtures (seeded RNGs, mocked scipy
when needed), and do not require a running LLM or DB.
"""
import math

import numpy as np
import pandas as pd
import pytest
import networkx as nx
import scipy.linalg as slin

from services.layer1 import notears as notears_mod
from services.layer1.notears import notears_linear, NotearsGraphBuilder


# ---------------------------------------------------------------------------
# BUG #4: _loss only handles 'l2'; any other loss type raises UnboundLocalError
# ===========================================================================
def test_notears_loss_handles_l2_and_l1_types(simple_causal_df):
    """Catches SWARM-3 BUG #4: notears_linear's `_loss` only handles 'l2'.
    Any other loss_type raises UnboundLocalError because `loss` and `G_loss`
    are never bound.

    Both 'l2' and 'l1' must work without error.
    """
    X = simple_causal_df.to_numpy(dtype=float)

    # l2 should work
    W_l2 = notears_linear(X, lambda1=0.05, loss_type='l2',
                          max_iter=5, w_threshold=0.0)
    assert W_l2.shape == (3, 3)
    assert np.all(np.isfinite(W_l2)), f"l2 W has non-finite values: {W_l2}"

    # l1 should also work (currently raises UnboundLocalError)
    W_l1 = notears_linear(X, lambda1=0.05, loss_type='l1',
                          max_iter=5, w_threshold=0.0)
    assert W_l1.shape == (3, 3)
    assert np.all(np.isfinite(W_l1)), f"l1 W has non-finite values: {W_l1}"


def test_notears_loss_unknown_type_raises_clear_error(simple_causal_df):
    """Catches SWARM-3 BUG #4: unknown loss_type raises UnboundLocalError
    instead of a clear ValueError. The error message should indicate that
    the loss_type is unsupported, not raise a bare NameError/UnboundLocalError.
    """
    X = simple_causal_df.to_numpy(dtype=float)

    # Unknown loss type — should raise ValueError, NOT UnboundLocalError
    with pytest.raises(ValueError, match="loss|unsupported|invalid|type"):
        notears_linear(X, lambda1=0.05, loss_type='unknown_type',
                       max_iter=5, w_threshold=0.0)


# ---------------------------------------------------------------------------
# BUG #18: notears_linear does not check sol.success from L-BFGS-B
# ===========================================================================
def test_notears_checks_sol_success(monkeypatch, simple_causal_df):
    """Catches SWARM-3 BUG #18: notears_linear does not check
    `sol.success` from scipy.optimize.minimize. If sol.success=False,
    the function silently uses sol.x (which may be garbage).

    The function should detect sol.success=False and raise an error
    (or log a clear warning), not silently return a meaningless W.
    """
    X = simple_causal_df.to_numpy(dtype=float)
    d = X.shape[1]

    # Mock scipy.optimize.minimize to return sol with success=False
    fake_sol = MagicMock()
    fake_sol.success = False
    fake_sol.x = np.full(2 * d * d, 0.5)  # garbage values
    fake_sol.message = "Mock convergence failure"

    monkeypatch.setattr(notears_mod.sopt, "minimize",
                        lambda *args, **kwargs: fake_sol)

    # The function should detect sol.success=False and raise an error
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        notears_linear(X, lambda1=0.05, max_iter=3, w_threshold=0.0)

    # The error message should mention convergence/success/optimization
    err_str = str(exc_info.value).lower()
    assert any(s in err_str for s in
               ["success", "converge", "optim", "fail", "minimize"]), (
        f"BUG #18: error message doesn't mention convergence failure: "
        f"{exc_info.value}"
    )


# ---------------------------------------------------------------------------
# BUG #19: first outer iteration always breaks the inner loop (h=inf init)
# ===========================================================================
def test_notears_first_iteration_does_not_always_break(monkeypatch):
    """Catches SWARM-3 BUG #19: notears_linear initializes `h = np.inf`.
    On the first outer iteration, the check `h_new > 0.25 * h` becomes
    `finite > inf` which is always False, so the inner loop always breaks
    immediately, never increasing rho on the first iteration.

    With the fix (e.g., h=0 init), rho should increase on the first
    iteration when h_new is large.
    """
    # Use random data where the unconstrained LS solution has h(W) > 0
    # (so rho should increase to enforce acyclicity)
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 4))

    # Track rho values across minimize calls by inspecting _func's closure
    rho_per_call = []
    original_minimize = notears_mod.sopt.minimize

    def tracking_minimize(func, x0, **kwargs):
        rho = None
        if hasattr(func, "__closure__") and func.__closure__:
            freevars = func.__code__.co_freevars
            if "rho" in freevars:
                idx = freevars.index("rho")
                val = func.__closure__[idx].cell_contents
                if isinstance(val, (int, float)):
                    rho = float(val)
        rho_per_call.append(rho)
        return original_minimize(func, x0, **kwargs)

    monkeypatch.setattr(notears_mod.sopt, "minimize", tracking_minimize)

    notears_linear(X, lambda1=0.05, max_iter=10, w_threshold=0.0)

    # We should have at least 2 minimize calls
    assert len(rho_per_call) >= 2, (
        f"Only {len(rho_per_call)} minimize calls — algorithm didn't run"
    )
    first_rho = rho_per_call[0]
    second_rho = rho_per_call[1]
    assert first_rho is not None and second_rho is not None, (
        f"Could not extract rho values from _func closure: {rho_per_call}"
    )

    # With BUG #19: first outer iter breaks immediately (rho stays at 1.0 for
    #   the first two calls — first call = outer 1, second call = outer 2 start
    #   since rho didn't grow in outer 1).
    # With the fix: first outer iter iterates the inner loop, so rho grows
    #   from 1.0 to 10.0 between the first and second minimize calls.
    assert second_rho > first_rho, (
        f"BUG #19: rho did not increase between first two minimize calls "
        f"({first_rho} → {second_rho}) — first outer iteration broke immediately "
        f"due to h=inf initialization"
    )


# ---------------------------------------------------------------------------
# BUG #17: _h uses scipy.linalg.expm which can overflow for large W
# ===========================================================================
def test_notears_acyclicity_constraint_finite():
    """Catches SWARM-3 BUG #17: notears_linear's `_h` uses
    `scipy.linalg.expm(W * W)` which can overflow for large W
    (when eigenvalues of W*W exceed ~700, expm returns inf).

    For moderate W (eigenvalues well under 700), h(W) must be finite.
    """
    d = 4
    # W with eigenvalues of W*W well under 700 (so expm doesn't overflow)
    rng = np.random.default_rng(42)
    W = rng.normal(0, 0.5, (d, d))
    np.fill_diagonal(W, 0)
    h = np.trace(slin.expm(W * W)) - d
    assert np.isfinite(h), f"BUG #17: h(W) not finite for moderate W: {h}"

    # Also test that notears_linear's output W has finite h
    X = rng.normal(0, 1, (50, d))
    W_out = notears_linear(X, lambda1=0.05, max_iter=10, w_threshold=0.0)
    h_out = np.trace(slin.expm(W_out * W_out)) - d
    assert np.isfinite(h_out), (
        f"BUG #17: notears output h(W) not finite: {h_out}"
    )


# ---------------------------------------------------------------------------
# BUG #6 (related): notears_linear reproducibility
# ===========================================================================
def test_notears_random_seed_is_set(simple_causal_df):
    """Catches SWARM-3 BUG #6 (related): notears_linear should be
    reproducible. Running twice on the same data must produce the same W
    (no unseeded randomness inside the optimization).
    """
    X = simple_causal_df.to_numpy(dtype=float)
    W1 = notears_linear(X, lambda1=0.05, max_iter=10, w_threshold=0.0)
    W2 = notears_linear(X, lambda1=0.05, max_iter=10, w_threshold=0.0)
    assert np.allclose(W1, W2), (
        f"BUG #6: notears_linear non-deterministic:\n{W1}\nvs\n{W2}"
    )


# ---------------------------------------------------------------------------
# Regression: notears output must be a DAG
# ===========================================================================
def test_notears_returns_dag_not_digraph(simple_causal_df):
    """Catches SWARM-3 (regression): notears_linear's output W must
    represent a DAG (no cycles). The acyclicity constraint
    `h(W) = tr(expm(W ∘ W)) - d` must enforce this.
    """
    X = simple_causal_df.to_numpy(dtype=float)
    W = notears_linear(X, lambda1=0.05, max_iter=20, w_threshold=0.3)

    G = nx.DiGraph()
    d = W.shape[0]
    G.add_nodes_from(range(d))
    for i in range(d):
        for j in range(d):
            if i != j and abs(W[i, j]) > 1e-10:
                G.add_edge(i, j)
    assert nx.is_directed_acyclic_graph(G), (
        f"BUG (regression): NOTEARS output is not a DAG; "
        f"cycles={list(nx.simple_cycles(G))}"
    )


# ---------------------------------------------------------------------------
# Regression: thresholding should not silently drop all edges
# ===========================================================================
def test_notears_threshold_does_not_silently_drop_all_edges(simple_causal_df):
    """Catches SWARM-3 (regression): when W has small coefficients,
    thresholding must not silently drop ALL edges. For data with a known
    causal structure (X→Y→Z), NOTEARS should produce at least one edge.
    """
    X = simple_causal_df.to_numpy(dtype=float)
    # Use a small threshold to preserve edges
    W = notears_linear(X, lambda1=0.01, max_iter=20, w_threshold=0.01)
    n_nonzero = int(np.sum(np.abs(W) > 0))
    assert n_nonzero > 0, (
        f"BUG (regression): NOTEARS threshold dropped all edges. W=\n{W}"
    )


# ---------------------------------------------------------------------------
# BUG #6 (related): singular covariance handling
# ===========================================================================
def test_notears_handles_singular_covariance(collinear_df):
    """Catches SWARM-3 BUG #6 (related): singular covariance matrix
    (perfectly collinear columns) should not crash notears_linear.

    notears_linear must handle singularity gracefully (e.g., via the
    +1e-8 std-dev floor) or raise a clear error — not crash with a
    numpy/scipy LinAlgError.
    """
    X = collinear_df.to_numpy(dtype=float)
    try:
        W = notears_linear(X, lambda1=0.05, max_iter=10, w_threshold=0.0)
        assert W.shape == (3, 3)
        assert np.all(np.isfinite(W)), f"W has non-finite values: {W}"
    except ValueError as e:
        # Acceptable: a clear error about singularity
        assert any(s in str(e).lower()
                   for s in ["singular", "ill-conditioned", "rank", "variance"]), (
            f"notears_linear raised unrelated ValueError: {e}"
        )
    except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError,
            KeyError, IndexError) as e:
        pytest.fail(
            f"BUG #6: notears_linear crashed on singular covariance with "
            f"{type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# Regression: max_iter parameter is respected
# ===========================================================================
def test_notears_max_iter_respected(monkeypatch, simple_causal_df):
    """Catches SWARM-3 (regression): notears_linear must respect the
    `max_iter` parameter (number of outer augmented-Lagrangian iterations).

    The algorithm should run at most max_iter outer iterations.
    """
    X = simple_causal_df.to_numpy(dtype=float)

    # Count calls to scipy.optimize.minimize
    minimize_calls = [0]
    original_minimize = notears_mod.sopt.minimize

    def counting_minimize(*args, **kwargs):
        minimize_calls[0] += 1
        return original_minimize(*args, **kwargs)

    monkeypatch.setattr(notears_mod.sopt, "minimize", counting_minimize)

    max_iter = 3
    notears_linear(X, lambda1=0.05, max_iter=max_iter, w_threshold=0.0)

    # Each outer iteration does at least 1 minimize call (so total >= max_iter)
    # Upper bound: each outer iter's inner loop multiplies rho by 10, at most
    # ~16 times before rho_max=1e16 is hit. So total <= max_iter * 20 (generous).
    assert minimize_calls[0] >= max_iter, (
        f"BUG (regression): minimize called only {minimize_calls[0]} times "
        f"for max_iter={max_iter} (should be >= {max_iter})"
    )
    upper_bound = max_iter * 50  # generous upper bound
    assert minimize_calls[0] <= upper_bound, (
        f"BUG (regression): minimize called {minimize_calls[0]} times for "
        f"max_iter={max_iter} — too many (max={upper_bound})"
    )


# ---------------------------------------------------------------------------
# BUG #22: confidence range
# ===========================================================================
def test_notears_confidence_in_zero_to_hundred_range(simple_causal_df):
    """Catches SWARM-3 BUG #22: NotearsGraphBuilder edge confidence must be
    in [0, 100], not in [0, |r|] (which is the correlation scale).

    A common fake fix is to use `abs(weight) * 100` without clamping,
    which can exceed 100 — this test catches that.
    """
    result = NotearsGraphBuilder.build_graph(simple_causal_df)
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


# ---------------------------------------------------------------------------
# BUG #6: unseeded jitter in NotearsGraphBuilder.build_graph
# ===========================================================================
def test_notears_does_not_use_unseeded_jitter(simple_causal_df, monkeypatch):
    """Catches SWARM-3 BUG #6: NotearsGraphBuilder.build_graph adds
    `np.random.normal(0, 1e-4, data_matrix.shape)` jitter without a seed,
    making the output non-deterministic.

    Running the builder twice on the same data must produce the same graph.
    """
    # Mock np.random.normal to track whether it's called with a seeded RNG
    normal_call_count = [0]
    seed_call_count = [0]
    rng_call_count = [0]

    original_normal = np.random.normal
    original_rng = np.random.default_rng
    original_seed = np.random.seed

    def tracking_normal(*args, **kwargs):
        normal_call_count[0] += 1
        return original_normal(*args, **kwargs)

    def tracking_rng(seed=None):
        rng_call_count[0] += 1
        return original_rng(seed)

    def tracking_seed(seed=None):
        seed_call_count[0] += 1
        return original_seed(seed)

    monkeypatch.setattr(np.random, "normal", tracking_normal)
    monkeypatch.setattr(np.random, "default_rng", tracking_rng)
    monkeypatch.setattr(np.random, "seed", tracking_seed)

    # Run twice — if unseeded, results will differ
    result1 = NotearsGraphBuilder.build_graph(simple_causal_df)
    result2 = NotearsGraphBuilder.build_graph(simple_causal_df)

    # If np.random.normal was called (jitter added), a seed must also be set
    if normal_call_count[0] > 0:
        assert rng_call_count[0] > 0 or seed_call_count[0] > 0, (
            f"BUG #6: np.random.normal called {normal_call_count[0]} times "
            f"without any seeding (default_rng calls: {rng_call_count[0]}, "
            f"seed calls: {seed_call_count[0]})"
        )

    # Reproducibility check: running twice must give the same edges
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
        f"BUG #6: NotearsGraphBuilder non-deterministic (unseeded jitter):\n"
        f"{e1}\nvs\n{e2}"
    )


# Local import for the mock used in test_notears_checks_sol_success
from unittest.mock import MagicMock  # noqa: E402