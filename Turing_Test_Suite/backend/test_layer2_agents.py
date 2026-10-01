"""
test_layer2_agents.py

Tests for `python-engine/services/layer2/agent_explorer.py`,
`python-engine/services/layer2/agent_exploiter.py`, and
`python-engine/services/layer2/agent_contrarian.py`.

Each test names the exact SWARM-4 BUG ID it targets (see Turing_Bug_Audit_Report.md,
Task ID: SWARM-4). Tests are deterministic (seeds set in the autouse fixture)
and mock any LLM calls — no LLM API keys required.
"""
import os
import sys
import time
import asyncio
import inspect
import warnings

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# ---------------------------------------------------------------------------
# Module path bootstrap — module-local, does not touch conftest.py
# ---------------------------------------------------------------------------
_PY_ENGINE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "upload", "Turing-main", "python-engine")
)
if _PY_ENGINE_DIR not in sys.path:
    sys.path.insert(0, _PY_ENGINE_DIR)

import random as _random  # noqa: E402
import numpy as np  # noqa: E402

from schemas.layer2 import SearchSpace, AgentProposal  # noqa: E402
from services.layer2.agent_explorer import AgentExplorer  # noqa: E402
from services.layer2.agent_exploiter import AgentExploiter  # noqa: E402
from services.layer2.agent_contrarian import AgentContrarian  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _deterministic_rng(monkeypatch):
    """Seed Python's `random` and NumPy's global RNG (and the GP env var) so
    any code path that *correctly* consumes these sources is reproducible
    (BUG #11, BUG #23).
    """
    monkeypatch.setenv("GP_RANDOM_STATE", "42")
    monkeypatch.setenv("AGENT_RANDOM_STATE", "42")
    np.random.seed(42)
    _random.seed(42)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    np.random.seed(None)
    _random.seed(None)


@pytest.fixture
def simple_domain():
    return {"x": SearchSpace(min=0.0, max=10.0)}


@pytest.fixture
def two_var_domain():
    return {
        "x": SearchSpace(min=0.0, max=10.0),
        "y": SearchSpace(min=-5.0, max=5.0),
    }


@pytest.fixture
def base_point_center():
    return {"x": 5.0, "y": 0.0}


@pytest.fixture
def history_with_yield():
    return [
        {"yield": 0.4, "values": {"x": 4.0, "y": -1.0}},
        {"yield": 0.7, "values": {"x": 7.0, "y": 1.0}},
        {"yield": 0.9, "values": {"x": 8.0, "y": 2.0}},
    ]


# ===========================================================================
# Helper: invoke an agent's propose() — sync or async
# ===========================================================================
def _run_propose(agent, *args, **kwargs):
    """Call `agent.propose(...)` whether it is sync or async, returning the
    AgentProposal.  If the agent is async, run it in a fresh event loop.
    """
    method = getattr(agent, "propose", None)
    if method is None:
        pytest.fail(f"Agent {type(agent).__name__} has no `propose` method.")
    if inspect.iscoroutinefunction(method):
        return asyncio.run(method(*args, **kwargs))
    return method(*args, **kwargs)


# ===========================================================================
# AGENT EXPLORER
# ===========================================================================
def test_explorer_returns_valid_proposal(base_point_center, two_var_domain):
    """Contract test — Explorer output must be an AgentProposal with the
    required keys.  The proposed_values dict must contain every node in
    domain_config.  An LLM that fakes a fix by returning `{}` will fail.
    """
    ex = AgentExplorer()
    proposal = _run_propose(ex, base_point_center, two_var_domain)

    assert isinstance(proposal, AgentProposal), (
        f"Explorer must return AgentProposal, got {type(proposal).__name__}"
    )
    assert proposal.agent_name, "agent_name must be non-empty"
    assert isinstance(proposal.proposed_values, dict), "proposed_values must be dict"
    assert isinstance(proposal.justification, str) and proposal.justification, (
        "justification must be a non-empty string"
    )
    # Must cover every node in domain_config
    assert set(proposal.proposed_values.keys()) == set(two_var_domain.keys()), (
        f"proposed_values must cover every node in domain_config. "
        f"Got: {set(proposal.proposed_values.keys())}, "
        f"Expected: {set(two_var_domain.keys())}"
    )
    # Values must be within bounds
    for node, space in two_var_domain.items():
        v = proposal.proposed_values[node]
        assert space.min <= v <= space.max, (
            f"Explorer proposed {v} for '{node}' which is outside "
            f"[{space.min}, {space.max}]"
        )


def test_explorer_async_uses_acompletion(base_point_center, two_var_domain):
    """SWARM-4 BUG #41 / BUG #42 (High) — async LLM calls must use
    `litellm.acompletion` (or similar async API), not sync `litellm.completion`.

    The current Explorer does NOT call the LLM at all (it's deterministic).
    This test is defensive: it mocks both `litellm.completion` and
    `litellm.acompletion` and asserts the SYNC `completion` was NOT invoked
    during `propose()`.  If a future bugfix adds an LLM-based path using
    sync `completion`, this test will catch it (the event loop would block).
    """
    # Build mock responses that look like a LiteLLM completion response
    sync_mock = MagicMock(return_value={"choices": [{"message": {"content": "{}"}}]})
    async_mock = AsyncMock(return_value={"choices": [{"message": {"content": "{}"}}]})

    with patch("litellm.completion", sync_mock), \
         patch("litellm.acompletion", async_mock):
        ex = AgentExplorer()
        _run_propose(ex, base_point_center, two_var_domain)

    # Sync `completion` must NOT have been called — it would block the event
    # loop if invoked from an async context.
    assert not sync_mock.called, (
        "AgentExplorer used sync `litellm.completion` — this blocks the "
        "event loop.  Use `litellm.acompletion` (async) instead.  "
        "(SWARM-4 BUG #41 / BUG #42)"
    )


def test_explorer_handles_llm_malformed_json(base_point_center, two_var_domain):
    """SWARM-4 BUG #43 (Medium) — must handle malformed JSON from the LLM
    (retry or fall back to a deterministic proposal).

    The current Explorer is deterministic and doesn't call the LLM, so
    this test passes trivially.  If a future bugfix adds an LLM call, this
    test verifies it doesn't crash on malformed JSON.  An LLM that fakes a
    fix by raising on bad JSON will fail.
    """
    # Mock LLM to return malformed JSON
    bad_response = MagicMock()
    bad_response.choices = [MagicMock()]
    bad_response.choices[0].message = MagicMock(content="not valid json {{")

    with patch("litellm.completion", MagicMock(return_value=bad_response)), \
         patch("litellm.acompletion", AsyncMock(return_value=bad_response)):
        ex = AgentExplorer()
        # Must not raise — must produce a valid AgentProposal (deterministic
        # fallback if the LLM is unavailable or returns garbage)
        try:
            proposal = _run_propose(ex, base_point_center, two_var_domain)
        except Exception as e:
            pytest.fail(
                f"Explorer crashed on malformed LLM JSON: {type(e).__name__}: {e}. "
                "SWARM-4 BUG #43: must handle JSON parse errors (retry or fallback)."
            )
        assert isinstance(proposal, AgentProposal)
        assert set(proposal.proposed_values.keys()) == set(two_var_domain.keys())


def test_explorer_respects_max_iter(base_point_center, two_var_domain):
    """SWARM-4 BUG #46 (Medium) — agent must not loop forever.

    The current Explorer is a single-shot `propose()` call — no loop.  If a
    future bugfix adds a retry loop (e.g. for LLM calls), it must be
    bounded.  This test mocks the LLM to ALWAYS fail and verifies the
    `propose()` call returns within a few seconds (not infinite loop).
    """
    call_count = {"n": 0}

    def _always_fail(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError("LLM always fails")

    with patch("litellm.completion", side_effect=_always_fail), \
         patch("litellm.acompletion", side_effect=_always_fail):
        ex = AgentExplorer()
        start = time.time()
        try:
            _run_propose(ex, base_point_center, two_var_domain)
        except Exception:
            pass  # Allowed to raise — the test is about bounded execution time.
        elapsed = time.time() - start

    assert elapsed < 5.0, (
        f"Explorer.propose() took {elapsed:.2f}s — possibly infinite retry loop "
        "(SWARM-4 BUG #46: LLM calls need a bounded retry count)."
    )
    # If the LLM was called at all, it must not be called unbounded times.
    # The current explorer calls the LLM 0 times (deterministic), so this is
    # 0.  A bugfix that adds retries must cap them.
    assert call_count["n"] <= 10, (
        f"LLM was called {call_count['n']} times — no max_iter bound on retries "
        "(SWARM-4 BUG #46)."
    )


def test_explorer_random_seed_reproducible(monkeypatch):
    """SWARM-4 BUG #23 (Medium) — same input + same RNG seed must give same
    output.  The buggy Explorer uses `random.uniform` (global RNG) when a
    node is missing from base_point, so two runs with DIFFERENT global RNG
    states give different proposals.  The fix must consume a local RNG
    seeded from `AGENT_RANDOM_STATE` (or similar).

    To make this test DETERMINISTIC (not flaky), we use a 10-variable
    domain with an empty base_point — every variable triggers the random
    fallback.  Each variable's random val determines its boundary choice
    (binary: min or max).  The probability that the buggy code (using the
    global RNG) produces IDENTICAL boundary choices across two runs with
    different global states is ~2^-10 = 0.1%, so the buggy code fails
    consistently.
    """
    monkeypatch.setenv("AGENT_RANDOM_STATE", "42")
    # 10 variables, all missing from base_point → all trigger random fallback
    domain = {f"v{i}": SearchSpace(min=0.0, max=10.0) for i in range(10)}
    empty_bp = {}

    # Run 1: advance the global RNG to a known distinct state
    _random.seed(999)
    for _ in range(1000):
        _random.uniform(0, 1)
    np.random.seed(999); _ = np.random.uniform(0, 1, 1000)
    ex1 = AgentExplorer()
    r1 = _run_propose(ex1, empty_bp, domain)

    # Run 2: DIFFERENT global RNG state
    _random.seed(7)
    for _ in range(5000):
        _random.uniform(0, 1)
    np.random.seed(7); _ = np.random.uniform(0, 1, 5000)
    ex2 = AgentExplorer()
    r2 = _run_propose(ex2, empty_bp, domain)

    assert r1.proposed_values == r2.proposed_values, (
        "Explorer is non-reproducible: two runs with same AGENT_RANDOM_STATE "
        "but different global RNG state produced different proposals. "
        f"({r1.proposed_values} vs {r2.proposed_values}). "
        "SWARM-4 BUG #23: use a local RNG seeded from env var, not global random."
    )


# ===========================================================================
# AGENT EXPLOITER
# ===========================================================================
def test_exploiter_returns_valid_proposal(history_with_yield, two_var_domain):
    """Contract test — Exploiter output must be an AgentProposal covering
    every node in domain_config.
    """
    ex = AgentExploiter()
    proposal = _run_propose(ex, history_with_yield, two_var_domain, sink_node="yield")

    assert isinstance(proposal, AgentProposal)
    assert proposal.agent_name
    assert set(proposal.proposed_values.keys()) == set(two_var_domain.keys())
    for node, space in two_var_domain.items():
        v = proposal.proposed_values[node]
        assert space.min <= v <= space.max


def test_exploiter_does_not_leave_proposed_unset(history_with_yield, two_var_domain, monkeypatch):
    """SWARM-4 BUG #19 (High) — `proposed[node]` must ALWAYS be set, even
    when neither nudge is in bounds.

    The buggy code has `if up_in_bounds and down_in_bounds: ...; elif
    up_in_bounds: ...; elif down_in_bounds: ...` with NO else.  When the
    historical best is near the center and `nudge_pct` is large (e.g. 0.6),
    BOTH up_val and down_val go out of bounds, so `proposed[node]` is never
    set, and the node is silently missing from the proposal.

    This test sets `EXPLOITER_NUDGE_PCT=0.6` and a history where the best
    is at the exact center, then asserts EVERY node is present in the
    proposal.  A fake fix that only handles the "both in bounds" case will
    fail.
    """
    monkeypatch.setenv("EXPLOITER_NUDGE_PCT", "0.6")
    # History where the best is at the exact center (5, 0) — nudge of
    # 0.6 * 10 = 6 puts up_val = 11 (OOB) and down_val = -1 (OOB) for x;
    # similarly for y.
    history = [
        {"yield": 0.5, "values": {"x": 5.0, "y": 0.0}},
        {"yield": 0.3, "values": {"x": 4.0, "y": -0.5}},
    ]
    ex = AgentExploiter()
    proposal = _run_propose(ex, history, two_var_domain, sink_node="yield")

    # EVERY node in domain_config must be in proposed_values — even when
    # neither nudge was in bounds, the agent must fall back to a boundary
    # clamp (or skip the nudge).
    for node in two_var_domain:
        assert node in proposal.proposed_values, (
            f"BUG #19: 'proposed' is missing node '{node}' — the nudge logic "
            "fell through with no else branch.  Add a fallback (e.g. clamp to "
            "the nearest boundary or keep the original best_val)."
        )
        # And the value must be in bounds
        v = proposal.proposed_values[node]
        space = two_var_domain[node]
        assert space.min <= v <= space.max, (
            f"BUG #19: proposed value {v} for '{node}' is out of bounds "
            f"[{space.min}, {space.max}] — fallback must clamp."
        )


def test_exploiter_nudge_within_bounds(history_with_yield, two_var_domain):
    """Contract test — nudged values must stay within bounds."""
    ex = AgentExploiter()
    proposal = _run_propose(ex, history_with_yield, two_var_domain, sink_node="yield")
    for node, space in two_var_domain.items():
        v = proposal.proposed_values[node]
        assert space.min <= v <= space.max, (
            f"Nudged value {v} for '{node}' is outside [{space.min}, {space.max}]"
        )


def test_exploiter_handles_bounds_edge_case(history_with_yield, two_var_domain, monkeypatch):
    """SWARM-4 BUG #19 (High) — when a nudge would push out of bounds, the
    value must be CLAMPED or the nudge REVERSED — NOT silently dropped.

    This test sets nudge_pct very high (0.95) so that for the historical
    best at the center, BOTH up_val and down_val go out of bounds.  The
    agent must NOT silently drop the node; it must clamp to a boundary.

    A fake fix that returns an empty proposal for this edge case will
    fail the "node present" assertion.
    """
    monkeypatch.setenv("EXPLOITER_NUDGE_PCT", "0.95")
    # History where the best is near center — nudge of 0.95*10 = 9.5 puts
    # up_val = 14.5 (OOB) and down_val = -4.5 (OOB) for x.
    history = [
        {"yield": 0.5, "values": {"x": 5.0, "y": 0.0}},
        {"yield": 0.1, "values": {"x": 1.0, "y": -3.0}},
    ]
    ex = AgentExploiter()
    proposal = _run_propose(ex, history, two_var_domain, sink_node="yield")

    # All nodes must be present (not silently dropped)
    assert set(proposal.proposed_values.keys()) == set(two_var_domain.keys()), (
        f"BUG #19: With nudge_pct=0.95, some nodes are missing from "
        f"proposed_values: got {set(proposal.proposed_values.keys())}, "
        f"expected {set(two_var_domain.keys())}.  The nudge logic needs a "
        "fallback branch (clamp or reverse the nudge)."
    )
    # Values must be within bounds (not just dropped)
    for node, space in two_var_domain.items():
        v = proposal.proposed_values[node]
        assert space.min <= v <= space.max, (
            f"BUG #19: Clamped value {v} for '{node}' is outside "
            f"[{space.min}, {space.max}]."
        )


# ===========================================================================
# AGENT CONTRARIAN
# ===========================================================================
def test_contrarian_returns_valid_proposal(base_point_center, two_var_domain):
    """Contract test — Contrarian output must be an AgentProposal covering
    every node in domain_config.
    """
    ct = AgentContrarian()
    proposal = _run_propose(ct, base_point_center, two_var_domain)

    assert isinstance(proposal, AgentProposal)
    assert proposal.agent_name
    assert set(proposal.proposed_values.keys()) == set(two_var_domain.keys())
    for node, space in two_var_domain.items():
        v = proposal.proposed_values[node]
        assert space.min <= v <= space.max


def test_contrarian_finds_counterexample(base_point_center, two_var_domain):
    """SWARM-4 BUG #24 (Medium) — the Contrarian MUST propose a DIFFERENT
    boundary from the Explorer for at least one node, providing a
    "counterexample" to the Explorer's choice.  This ensures no two agents
    test the same corner in the same round (the docstring promise).

    With base_point at center {(5, 0)}, the Explorer picks the FARTHER
    boundary per node.  The Contrarian must pick the OPPOSITE (closer)
    boundary — so for each node, their proposals should DIFFER.

    A fake fix that makes the Contrarian always agree with the Explorer
    (e.g. always pick the same boundary) will fail this test.
    """
    ex = AgentExplorer()
    ct = AgentContrarian()
    explorer_prop = _run_propose(ex, base_point_center, two_var_domain)
    contrarian_prop = _run_propose(ct, base_point_center, two_var_domain)

    # The Contrarian must propose DIFFERENT values from the Explorer for
    # at least one node (anti-duplication guarantee).
    differences = [
        node for node in two_var_domain
        if explorer_prop.proposed_values[node] != contrarian_prop.proposed_values[node]
    ]
    assert len(differences) >= 1, (
        f"BUG #24: Contrarian and Explorer proposed IDENTICAL values for "
        f"every node ({explorer_prop.proposed_values}).  The Contrarian must "
        "challenge the Explorer by picking the opposite boundary — otherwise "
        "two of three simulations per round are wasted on the same corner."
    )


def test_contrarian_does_not_always_disagree(base_point_center, two_var_domain):
    """Contract / fake-fix detection — the Contrarian must NOT disagree for
    the SAKE of disagreeing in a way that ignores the proposal's correctness.

    Concretely: when the base_point is at the EXACT center of every
    SearchSpace, the Explorer and Contrarian BOTH pick a boundary (Explorer
    picks MIN per the documented tie-break, Contrarian picks MAX).  The
    Contrarian's choice must be a valid value within bounds — it must not
    produce NaN, None, or an out-of-bounds value just to "disagree".

    An LLM that fakes the Contrarian by always returning `disagree` without
    producing a valid proposal will fail the bounds check.
    """
    ct = AgentContrarian()
    proposal = _run_propose(ct, base_point_center, two_var_domain)

    # All proposed values must be in bounds — disagreeing doesn't mean
    # producing garbage.
    for node, space in two_var_domain.items():
        v = proposal.proposed_values[node]
        assert v is not None, f"Contrarian returned None for '{node}'"
        assert isinstance(v, (int, float)), (
            f"Contrarian returned {type(v).__name__} for '{node}', expected number"
        )
        assert space.min <= v <= space.max, (
            f"Contrarian value {v} for '{node}' is outside [{space.min}, {space.max}]"
        )
        assert not (isinstance(v, float) and (v != v)), (
            f"Contrarian returned NaN for '{node}'"
        )


# ===========================================================================
# ALL AGENTS — STATE ISOLATION
# ===========================================================================
@pytest.mark.parametrize("AgentClass,make_args", [
    (AgentExplorer, lambda: ({"x": 5.0, "y": 0.0}, {"x": SearchSpace(min=0, max=10), "y": SearchSpace(min=-5, max=5)})),
    (AgentContrarian, lambda: ({"x": 5.0, "y": 0.0}, {"x": SearchSpace(min=0, max=10), "y": SearchSpace(min=-5, max=5)})),
    (AgentExploiter, lambda: (
        [{"yield": 0.5, "values": {"x": 5.0, "y": 0.0}}],
        {"x": SearchSpace(min=0, max=10), "y": SearchSpace(min=-5, max=5)},
    )),
])
def test_all_agents_state_not_mutated_between_calls(AgentClass, make_args):
    """SWARM-4 BUG #23 (Medium) — calling an agent twice must not leak state
    from call 1 into call 2.

    The current agents are stateless (no __init__ state) so this passes
    trivially with seeded RNG.  A bugfix that adds instance-level mutable
    state (e.g. a cache) without proper isolation will fail this test.
    """
    args = make_args()
    agent1 = AgentClass()
    agent2 = AgentClass()

    # Reset global RNG before each call so the only source of variation
    # would be agent-internal state.
    _random.seed(42); np.random.seed(42)
    r1 = _run_propose(agent1, *args, sink_node="yield") if AgentClass is AgentExploiter \
         else _run_propose(agent1, *args)

    _random.seed(42); np.random.seed(42)
    r2 = _run_propose(agent2, *args, sink_node="yield") if AgentClass is AgentExploiter \
         else _run_propose(agent2, *args)

    assert r1.proposed_values == r2.proposed_values, (
        f"{AgentClass.__name__} produced different outputs on two fresh "
        f"instances with the same RNG seed — state is leaking between calls "
        "or the agent is using a non-deterministic source. "
        f"({r1.proposed_values} vs {r2.proposed_values})"
    )

    # Calling the SAME agent twice with the same seed must also produce
    # the same output (no internal state accumulation).
    _random.seed(42); np.random.seed(42)
    r3 = _run_propose(agent1, *args, sink_node="yield") if AgentClass is AgentExploiter \
         else _run_propose(agent1, *args)
    assert r1.proposed_values == r3.proposed_values, (
        f"{AgentClass.__name__} produced different outputs on repeated calls "
        f"to the same instance — internal state is accumulating."
    )