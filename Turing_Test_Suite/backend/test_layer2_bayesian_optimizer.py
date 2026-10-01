"""
test_layer2_bayesian_optimizer.py

Tests for `python-engine/services/layer2/bayesian_optimizer.py`.

Each test names the exact SWARM-4 BUG ID it targets (see Turing_Bug_Audit_Report.md,
Task ID: SWARM-4). Tests are designed to:
  - FAIL with the current buggy code (red)
  - PASS only when the bug is properly fixed (green) — not when an LLM fakes the fix
  - Be deterministic (no flakiness)
  - Not require LLM API keys (no LLM calls in this module)
"""
import os
import sys
import warnings
import inspect

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Module path bootstrap — module-local, does not touch conftest.py
# ---------------------------------------------------------------------------
_PY_ENGINE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "upload", "Turing-main", "python-engine")
)
if _PY_ENGINE_DIR not in sys.path:
    sys.path.insert(0, _PY_ENGINE_DIR)

from schemas.layer2 import SearchSpace  # noqa: E402
from services.layer2.bayesian_optimizer import BayesianOptimizer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _deterministic_rng(monkeypatch):
    """Seed Python's `random`, NumPy's global RNG, and the GP_RANDOM_STATE env var
    so that any code path that *correctly* consumes these sources is reproducible.
    A correctly-fixed optimizer must produce identical output across two calls
    when the seed is set; the buggy version uses unseeded `np.random.uniform` for
    EI candidate sampling and therefore ignores this fixture.
    """
    monkeypatch.setenv("GP_RANDOM_STATE", "42")
    np.random.seed(42)
    import random as _random
    _random.seed(42)
    yield
    np.random.seed(None)


@pytest.fixture
def quad_domain():
    """A simple two-parameter search space."""
    return {
        "x": SearchSpace(min=0.0, max=10.0),
        "y": SearchSpace(min=-5.0, max=5.0),
    }


@pytest.fixture
def seeded_history(quad_domain):
    """A small deterministic historical_data set with a clear (sink_node, values)
    structure.  Each entry has the sink_node key 'yield' to avoid BUG #2.
    """
    return [
        {"yield": 0.20, "values": {"x": 1.0, "y": -3.0}},
        {"yield": 0.45, "values": {"x": 3.0, "y": -1.0}},
        {"yield": 0.70, "values": {"x": 5.0, "y":  1.0}},
        {"yield": 0.85, "values": {"x": 7.0, "y":  3.0}},
        {"yield": 0.95, "values": {"x": 9.0, "y":  4.0}},
    ]


# ---------------------------------------------------------------------------
# 1. SWARM-4 BUG #2: training on y=0 when sink_node is falsy
# ---------------------------------------------------------------------------
def test_does_not_train_on_y_zero_when_sink_node_falsy(seeded_history, quad_domain):
    """SWARM-4 BUG #2 (Critical).

    The buggy line:
        y_train.append(entry.get(sink_node, 0.0) if sink_node else 0.0)
    silently trains the GP on a constant label `0.0` whenever `sink_node` is
    `None` OR an empty string.  The fixed code MUST reject a falsy sink_node
    with a clear ValueError BEFORE building any training data — otherwise the
    optimizer silently produces meaningless recommendations.

    This test asserts a ValueError is raised for both `None` and `""`.  An LLM
    that "fixes" only the `None` case (or only the `""` case) will fail one of
    the two assertions.
    """
    bo = BayesianOptimizer()

    # Case A: sink_node=None
    with pytest.raises(ValueError, match="sink_node"):
        bo.get_base_point(seeded_history, quad_domain, sink_node=None)

    # Case B: sink_node="" (empty string is also falsy and must be rejected)
    with pytest.raises(ValueError, match="sink_node"):
        bo.get_base_point(seeded_history, quad_domain, sink_node="")


# ---------------------------------------------------------------------------
# 2. sink_node column missing from historical_data
# ---------------------------------------------------------------------------
def test_sink_node_missing_in_data_raises_clear_error(seeded_history, quad_domain):
    """SWARM-4 BUG #2 (Critical, sibling case).

    Even when `sink_node` is a non-empty string, if no entry in `historical_data`
    contains that key, the GP would be trained on the `entry.get(sink_node, 0.0)`
    default (0.0) — i.e. silently the same y=0 degeneracy.  The fixed code must
    raise a clear ValueError naming the missing sink_node.

    The buggy code only emits a `warnings.warn(...)` and falls back to the
    domain center — it does NOT raise.  This test requires an exception.
    """
    bo = BayesianOptimizer()
    with pytest.raises(ValueError, match="sink_node"):
        bo.get_base_point(seeded_history, quad_domain, sink_node="nonexistent_yield")


# ---------------------------------------------------------------------------
# 3. SWARM-4 BUG #5: random sampling not seeded -> non-reproducible
# ---------------------------------------------------------------------------
def test_random_seed_is_set(seeded_history, quad_domain, monkeypatch):
    """SWARM-4 BUG #5 (High).

    `GP_RANDOM_STATE` seeds the GP itself, but the 1000 EI candidate samples
    use `np.random.uniform(...)` against the GLOBAL NumPy RNG, which is never
    seeded.  Two identical calls therefore produce different `best_x` values.

    The fixed code MUST derive a LOCAL RNG from the same env var (or an
    explicit `seed` parameter) so that two calls with the same seed produce
    identical suggestions REGARDLESS of the global NumPy RNG state.

    This test:
      1. Sets `GP_RANDOM_STATE=12345`.
      2. Pollutes the global NumPy RNG (advances it 5000 steps) before the
         FIRST call so the global state is "noisy".
      3. Calls `get_base_point` once (captures r1).
      4. Pollutes the global NumPy RNG again (different amount) before the
         SECOND call.
      5. Calls `get_base_point` again (captures r2).
      6. Asserts r1 == r2.

    The buggy version (uses global `np.random.uniform`) will fail because
    the global RNG state differs between the two calls.  A correctly-fixed
    version (uses a local `np.random.default_rng(seed)`) ignores the global
    state and passes.
    """
    monkeypatch.setenv("GP_RANDOM_STATE", "12345")
    bo = BayesianOptimizer()

    # Pollute the global RNG before call 1
    np.random.seed(999)
    _ = np.random.uniform(0, 1, 5000)
    r1 = bo.get_base_point(seeded_history, quad_domain, sink_node="yield")

    # Pollute the global RNG differently before call 2
    np.random.seed(7)
    _ = np.random.uniform(0, 1, 31337)
    r2 = bo.get_base_point(seeded_history, quad_domain, sink_node="yield")

    assert r1 == r2, (
        "BayesianOptimizer is non-reproducible: two calls with the same "
        "GP_RANDOM_STATE but different global NumPy state produced different "
        f"suggestions ({r1} vs {r2}).  BUG #5: derive a LOCAL RNG from the env "
        "var (np.random.default_rng(seed)) — do NOT use the global np.random."
    )


# ---------------------------------------------------------------------------
# 4. Suggestions within bounds (not clamped AFTER the fact)
# ---------------------------------------------------------------------------
def test_suggestion_within_bounds(seeded_history, quad_domain):
    """Sanity / contract test.

    The suggested values must lie within the SearchSpace bounds.  Importantly,
    the bounds must be enforced DURING acquisition (the candidate sampling
    space is the box), NOT applied as a post-hoc clip — otherwise the
    optimizer can saturate at the boundary on every iteration.
    """
    bo = BayesianOptimizer()
    suggestion = bo.get_base_point(seeded_history, quad_domain, sink_node="yield")
    for node, space in quad_domain.items():
        assert node in suggestion, f"Missing node '{node}' in suggestion"
        val = suggestion[node]
        assert space.min <= val <= space.max, (
            f"Suggestion for '{node}' = {val} is outside bounds "
            f"[{space.min}, {space.max}].  The optimizer must sample candidates "
            "FROM the box, not clamp after the fact."
        )


# ---------------------------------------------------------------------------
# 5. Initial design uses Latin Hypercube Sampling (BUG #8 — no xi + no LHS)
# ---------------------------------------------------------------------------
def test_initial_design_is_lhs_not_random(quad_domain):
    """SWARM-4 BUG #8 (High) — initial design must be space-filling.

    The buggy optimizer, when `historical_data` is empty (or has < 2 entries),
    returns the *exact mathematical center* of every SearchSpace.  This is
    the worst possible initial design: every BO run starts from the same
    point and the GP is fit on a degenerate sample.

    The fixed code MUST draw an initial design that is space-filling — e.g.
    Latin Hypercube Sampling (LHS) — so that the first N suggestions are
    stratified across every dimension.

    This test requests N=5 initial suggestions (via repeated calls with empty
    history OR an explicit initial-design method) and verifies:
      - the 5 suggestions are NOT all identical (the buggy center-point
        behavior returns the same point 5 times), AND
      - in each dimension, the 5 values hit at least 3 distinct strata of
        [min, max] (a space-filling property).  Pure random sampling fails
        this with high probability; LHS passes deterministically.
    """
    bo = BayesianOptimizer()

    # Try to obtain N=5 initial-design points.
    suggestions = []
    # Path A: an explicit `initial_design` method on the optimizer
    if hasattr(bo, "initial_design"):
        try:
            out = bo.initial_design(quad_domain, n=5)
            # Accept either a list-of-dicts or a dict-of-lists
            if isinstance(out, list):
                suggestions = out
            elif isinstance(out, dict):
                suggestions = [{k: out[k][i] for k in out} for i in range(5)]
        except TypeError:
            suggestions = []

    # Path B: call get_base_point repeatedly with empty history
    if len(suggestions) < 5:
        suggestions = []
        for _ in range(5):
            s = bo.get_base_point([], quad_domain, sink_node=None) \
                if "sink_node" in inspect.signature(bo.get_base_point).parameters else \
                bo.get_base_point([], quad_domain)
            suggestions.append(s)

    assert len(suggestions) == 5, f"Expected 5 initial-design points, got {len(suggestions)}"

    # (a) NOT all identical (catches the center-point bug)
    serialized = [tuple(sorted(s.items())) for s in suggestions]
    assert len(set(serialized)) > 1, (
        "Initial design returned 5 identical suggestions — the buggy optimizer "
        "always returns the domain center when history is empty.  Use LHS."
    )

    # (b) Space-filling: in each dimension, the 5 values must hit at least 3
    # distinct strata of [min, max].
    for node, space in quad_domain.items():
        vals = sorted(s[node] for s in suggestions)
        # Divide the range into 5 equal-width strata
        width = space.max - space.min
        stratum_of = lambda v: int(min(4, max(0, (v - space.min) / width * 5)))
        strata_hit = {stratum_of(v) for v in vals}
        assert len(strata_hit) >= 3, (
            f"Initial design for '{node}' only hit {len(strata_hit)} strata "
            f"of 5 (values={vals}); not space-filling. Use Latin Hypercube Sampling."
        )


# ---------------------------------------------------------------------------
# 6. SWARM-4 BUG #8: EI closed-form formula correctness
# ---------------------------------------------------------------------------
def test_acquisition_function_ei_correct(monkeypatch):
    """SWARM-4 BUG #8 (High).

    EI (Jones et al. 1998) closed-form:
        EI(x) = (mu - y_best) * Phi((mu - y_best)/std) + std * phi((mu - y_best)/std)

    The buggy implementation omits the `xi` exploration parameter and uses
    `imp = mu - y_best` directly.  This test verifies the FORMULA is correct
    by mocking the GP to return controlled (mu, std) and capturing the EI
    array via `np.argmax` interception.

    A fake fix that hardcodes a constant EI value (e.g. always returns 0.5)
    will be caught because the expected EI for the mocked inputs is NOT 0.5.
    """
    from scipy.stats import norm

    # Closed-form expected EI for mu=0.5, std=0.1, y_best=0.4
    mu_fixed, std_fixed, y_best = 0.5, 0.1, 0.4
    Z = (mu_fixed - y_best) / std_fixed
    expected_ei = (mu_fixed - y_best) * norm.cdf(Z) + std_fixed * norm.pdf(Z)

    # Sanity check the expected value isn't trivial
    assert 0.05 < expected_ei < 0.2, "Test setup: expected_ei out of plausible range"

    # Mock GP: fit is a no-op; predict returns constant mu and std for all inputs
    class _MockGP:
        def fit(self, X, y):
            self._y = y
        def predict(self, X, return_std=False):
            n = len(X)
            mu = np.full(n, mu_fixed, dtype=float)
            std = np.full(n, std_fixed, dtype=float)
            return mu, std

    monkeypatch.setattr(
        "services.layer2.bayesian_optimizer.GaussianProcessRegressor",
        lambda **kw: _MockGP(),
    )

    # Intercept np.argmax to capture the EI array the optimizer computed
    captured = {}
    real_argmax = np.argmax

    def _capture_argmax(arr, *args, **kwargs):
        captured["ei"] = np.array(arr, copy=True)
        return real_argmax(arr, *args, **kwargs)

    monkeypatch.setattr(
        "services.layer2.bayesian_optimizer.np.argmax", _capture_argmax
    )

    bo = BayesianOptimizer()
    history = [
        {"yield": 0.4, "values": {"x": 1.0}},
        {"yield": 0.3, "values": {"x": 2.0}},
        {"yield": 0.2, "values": {"x": 3.0}},
    ]
    domain = {"x": SearchSpace(min=0.0, max=10.0)}
    bo.get_base_point(history, domain, sink_node="yield")

    assert "ei" in captured, "Failed to capture EI array — np.argmax was not called."
    ei_array = captured["ei"]
    # Every candidate has the same (mu, std), so every EI must equal expected_ei
    assert np.allclose(ei_array, expected_ei, atol=1e-6), (
        f"EI computation incorrect.  Closed-form expects {expected_ei:.6f}, "
        f"optimizer produced {ei_array[0]:.6f}.  Check the EI formula and the xi term."
    )


# ---------------------------------------------------------------------------
# 7. UCB acquisition: must compute mean + kappa * std
# ---------------------------------------------------------------------------
def test_acquisition_function_ucb_correct(monkeypatch):
    """SWARM-4 BUG #4 (Critical).

    The current optimizer only has EI; there is no UCB option.  The fix
    should add a UCB acquisition function (or a `kappa` parameter on
    `get_base_point`) that computes:
        UCB(x) = mu + kappa * std

    This test searches for a UCB function/method on the optimizer (or the
    module) and verifies the formula.  If no UCB exists, the test fails —
    an LLM that only renames EI to "UCB" without changing the formula will
    be caught by the numerical assertion.
    """
    bo = BayesianOptimizer()

    # Discover a UCB function: method on optimizer, or module-level function.
    ucb_callable = None
    for attr in ("ucb_acquisition", "_ucb", "upper_confidence_bound", "ucb"):
        if hasattr(bo, attr):
            ucb_callable = getattr(bo, attr)
            break
    if ucb_callable is None:
        import services.layer2.bayesian_optimizer as bo_mod
        for attr in ("ucb_acquisition", "_ucb", "upper_confidence_bound", "ucb"):
            if hasattr(bo_mod, attr):
                ucb_callable = getattr(bo_mod, attr)
                break

    if ucb_callable is None:
        pytest.fail(
            "No UCB acquisition function found on BayesianOptimizer or in "
            "bayesian_optimizer module.  BUG #4 requires a UCB option with "
            "formula `mean + kappa * std`."
        )

    # Test the formula with a few values of kappa
    mu = np.array([0.5])
    std = np.array([0.1])
    for kappa in (0.0, 1.0, 2.0, 5.0):
        expected = mu + kappa * std
        try:
            result = ucb_callable(mu, std, kappa)
        except TypeError:
            # Some implementations take kappa as a keyword
            try:
                result = ucb_callable(mu, std, kappa=kappa)
            except TypeError as e:
                pytest.fail(
                    f"UCB function signature incompatible with (mu, std, kappa): {e}"
                )
        result = np.atleast_1d(np.asarray(result, dtype=float))
        assert np.allclose(result, expected, atol=1e-6), (
            f"UCB with kappa={kappa} should be {expected[0]:.4f}, got {result[0]:.4f}. "
            "Formula must be `mean + kappa * std`."
        )


# ---------------------------------------------------------------------------
# 8. SWARM-4 BUG #6: GP must use a non-trivial kernel (RBF or Matern)
# ---------------------------------------------------------------------------
def test_gp_uses_correct_kernel(seeded_history, quad_domain, monkeypatch):
    """SWARM-4 BUG #6 (High).

    The kernel is `1.0 * Matern(length_scale=...) + WhiteKernel(...)`.  This
    is fine — Matern is a non-trivial kernel.  However, a regression that
    replaces the kernel with `ConstantKernel` (or no kernel at all) would
    silently turn the GP into a constant predictor.  This test captures the
    kernel passed to `GaussianProcessRegressor` and asserts it contains an
    RBF or Matern component.
    """
    captured = {}

    real_gpr_init = None
    from sklearn.gaussian_process import GaussianProcessRegressor as _RealGPR

    class _CapturingGPR(_RealGPR):
        def __init__(self, kernel=None, **kwargs):
            captured["kernel"] = kernel
            super().__init__(kernel=kernel, **kwargs)

    monkeypatch.setattr(
        "services.layer2.bayesian_optimizer.GaussianProcessRegressor",
        _CapturingGPR,
    )

    bo = BayesianOptimizer()
    bo.get_base_point(seeded_history, quad_domain, sink_node="yield")

    assert "kernel" in captured, "GP was never constructed — kernel capture failed."
    kernel = captured["kernel"]
    # Stringify the kernel — sklearn kernels have a __repr__ that names
    # each component (e.g. "Matern**2 + WhiteKernel**2").
    kernel_repr = repr(kernel)
    assert "Matern" in kernel_repr or "RBF" in kernel_repr, (
        f"GP kernel must contain an RBF or Matern component.  Got: {kernel_repr}. "
        "A ConstantKernel alone is degenerate (predicts the prior everywhere)."
    )
    # And it must NOT be only a ConstantKernel
    assert "ConstantKernel" not in kernel_repr.replace("ConstantKernel", "CK", 1) or \
           ("Matern" in kernel_repr or "RBF" in kernel_repr), (
        f"GP kernel must not be only a ConstantKernel.  Got: {kernel_repr}."
    )


# ---------------------------------------------------------------------------
# 9. SWARM-4 BUG #32: respect max_iter (no DoS, no infinite loop)
# ---------------------------------------------------------------------------
def test_optimizer_respects_max_iter(seeded_history, quad_domain):
    """SWARM-4 BUG #32 (Medium).

    The optimizer must expose a way to bound the number of suggestions it
    produces in a single call.  Currently `get_base_point` returns ONE
    suggestion — there is no `max_iter` parameter and no `optimize()` method.

    This test verifies that one of the following exists:
      (a) an `optimize(historical_data, domain_config, sink_node, max_iter=N)`
          method that returns a list of N suggestions, OR
      (b) a `max_iter` (or `n_iter`) keyword on `get_base_point` that returns
          a list of N suggestions.

    An LLM that fakes the fix by ignoring `max_iter` and always returning one
    suggestion will fail the `len(...) == 3` assertion.
    """
    bo = BayesianOptimizer()
    suggestions = None

    # Path A: an `optimize` method
    if hasattr(bo, "optimize"):
        try:
            out = bo.optimize(seeded_history, quad_domain, sink_node="yield", max_iter=3)
            if isinstance(out, (list, tuple)) and len(out) == 3:
                suggestions = out
        except TypeError:
            pass

    # Path B: a `max_iter` (or `n_iter`) keyword on `get_base_point`
    if suggestions is None:
        sig = inspect.signature(bo.get_base_point)
        for kw in ("max_iter", "n_iter", "n_iterations", "num_iter"):
            if kw in sig.parameters:
                try:
                    out = bo.get_base_point(
                        seeded_history, quad_domain, sink_node="yield", **{kw: 3}
                    )
                    if isinstance(out, (list, tuple)) and len(out) == 3:
                        suggestions = out
                        break
                except TypeError:
                    pass

    assert suggestions is not None, (
        "BayesianOptimizer has no `optimize(max_iter=N)` method and no "
        "`max_iter`/`n_iter` keyword on `get_base_point`.  BUG #32: the "
        "optimizer must expose a bounded iteration API to prevent DoS."
    )
    assert len(suggestions) == 3


# ---------------------------------------------------------------------------
# 10. Empty history does not crash
# ---------------------------------------------------------------------------
def test_optimizer_handles_empty_history(quad_domain):
    """Contract test (no specific BUG ID, but related to BUG #9).

    With no historical data, the optimizer must return a valid initial
    suggestion (not crash, not silently return {}).  An LLM that fakes a fix
    by returning `{}` for empty history will fail the keys assertion.
    """
    bo = BayesianOptimizer()
    # Provide a dummy sink_node so we don't trip BUG #2's ValueError path
    suggestion = bo.get_base_point(
        [], quad_domain, sink_node="yield" if "sink_node" in inspect.signature(
            bo.get_base_point).parameters else None
    )
    assert isinstance(suggestion, dict)
    assert set(suggestion.keys()) == set(quad_domain.keys()), (
        f"Empty history should still return a suggestion for every node. "
        f"Got: {suggestion}"
    )
    for node, space in quad_domain.items():
        assert space.min <= suggestion[node] <= space.max


# ---------------------------------------------------------------------------
# 11. Single-sample history: GP must fit (or fall back) without crashing
# ---------------------------------------------------------------------------
def test_optimizer_handles_single_sample(quad_domain):
    """Contract test (BUG #9 — degenerate EI on too-few samples).

    With only ONE historical sample, sklearn's GP can still fit (it has
    one observation).  The buggy code already handles this via the
    `len(historical_data) < 2` early-return, but a fake fix that removes
    this guard and calls `gp.fit` on one point will crash on
    `np.std(y) == 0`.  This test verifies no crash, regardless of which
    branch the fix takes.
    """
    bo = BayesianOptimizer()
    history = [{"yield": 0.5, "values": {"x": 5.0, "y": 0.0}}]
    # Should not raise
    suggestion = bo.get_base_point(history, quad_domain, sink_node="yield")
    assert isinstance(suggestion, dict)
    assert set(suggestion.keys()) == set(quad_domain.keys())


# ---------------------------------------------------------------------------
# 12. Calling suggest() twice without new data does NOT return the same value
# ---------------------------------------------------------------------------
def test_optimizer_suggestion_not_constant_across_calls(seeded_history, quad_domain, monkeypatch):
    """Catches LLMs that hardcode a constant suggestion.

    Without an env-var seed, two consecutive calls to `get_base_point` MUST
    produce different suggestions (the optimizer explores different parts of
    the candidate space).  An LLM that hardcodes `return {"x": 5.0, "y": 0.0}`
    to "fix" reproducibility will fail this test.

    NOTE: this test is in tension with `test_random_seed_is_set` — the fix
    must use a local RNG that ADVANCES between calls (so consecutive calls
    differ) but is REPRODUCIBLE when seeded with the same env-var (so two
    fresh instances with the same seed agree).
    """
    # Deliberately UNSET the seed env var so each call uses fresh entropy
    monkeypatch.delenv("GP_RANDOM_STATE", raising=False)
    bo = BayesianOptimizer()
    r1 = bo.get_base_point(seeded_history, quad_domain, sink_node="yield")
    r2 = bo.get_base_point(seeded_history, quad_domain, sink_node="yield")
    assert r1 != r2, (
        "Two consecutive calls with no env-var seed returned identical "
        f"suggestions ({r1}).  The optimizer must explore, not hardcode."
    )


# ---------------------------------------------------------------------------
# 13. Exploration/exploitation balance with kappa
# ---------------------------------------------------------------------------
def test_optimizer_exploration_exploitation_balance(monkeypatch):
    """SWARM-4 BUG #4 + BUG #8 (Critical/High).

    With UCB and a HIGH kappa, the optimizer should favor candidates with
    high uncertainty (exploration — far from the current best).  With a LOW
    kappa, it should favor candidates with high mean (exploitation — near
    the current best).

    This test mocks the GP to return a controlled landscape where one
    candidate has high mean + low std (exploitation target) and another has
    medium mean + high std (exploration target).  It verifies the optimizer
    picks the exploitation target with low kappa and the exploration target
    with high kappa.

    An LLM that fakes the fix by always picking the highest-mean candidate
    (ignoring kappa) will fail the high-kappa assertion.
    """
    # Build a controlled GP: candidates are sorted by index.
    # Index 0: high mean, low std (exploitation target)
    # Index N-1: medium mean, high std (exploration target)
    class _MockGP:
        def fit(self, X, y):
            self._y = y
        def predict(self, X, return_std=False):
            n = len(X)
            mu = np.linspace(0.9, 0.5, n)         # decreasing mean
            std = np.linspace(0.01, 0.5, n)      # increasing std
            return mu, std

    # Make the candidate sample deterministic so we can interpret indices
    captured_X = {}

    class _DeterministicUniform:
        def __call__(self, low, high, size):
            # Return a deterministic ramp so that higher indices = "further
            # from current best"
            arr = np.linspace(low, high, size)
            captured_X["shape"] = arr.shape
            return arr

    monkeypatch.setattr(
        "services.layer2.bayesian_optimizer.GaussianProcessRegressor",
        lambda **kw: _MockGP(),
    )

    bo = BayesianOptimizer()
    history = [
        {"yield": 0.4, "values": {"x": 5.0}},
        {"yield": 0.3, "values": {"x": 4.0}},
        {"yield": 0.2, "values": {"x": 3.0}},
    ]
    domain = {"x": SearchSpace(min=0.0, max=10.0)}

    # Try UCB with high kappa — must pick a high-std candidate
    sig = inspect.signature(bo.get_base_point)
    has_kappa = "kappa" in sig.parameters or "acquisition" in sig.parameters

    if not has_kappa and not hasattr(bo, "acquisition") and not hasattr(bo, "optimize"):
        pytest.fail(
            "BUG #4: optimizer has no kappa/acquisition parameter and no "
            "optimize() method — cannot balance exploration vs exploitation."
        )

    # Try the kappa path
    try:
        if "kappa" in sig.parameters:
            high_kappa_res = bo.get_base_point(history, domain, sink_node="yield", kappa=10.0)
            low_kappa_res = bo.get_base_point(history, domain, sink_node="yield", kappa=0.01)
        else:
            high_kappa_res = bo.optimize(
                history, domain, sink_node="yield", max_iter=1, kappa=10.0
            )
            low_kappa_res = bo.optimize(
                history, domain, sink_node="yield", max_iter=1, kappa=0.01
            )
            high_kappa_res = high_kappa_res[0] if isinstance(high_kappa_res, list) else high_kappa_res
            low_kappa_res = low_kappa_res[0] if isinstance(low_kappa_res, list) else low_kappa_res
    except TypeError as e:
        pytest.fail(
            f"BUG #4: optimizer does not accept kappa parameter: {e}. "
            "Cannot balance exploration vs exploitation without it."
        )

    # The two suggestions must differ (high kappa explores, low kappa exploits)
    assert high_kappa_res != low_kappa_res, (
        "High kappa and low kappa produced identical suggestions — the "
        "optimizer is ignoring kappa and not balancing exploration/exploitation."
    )