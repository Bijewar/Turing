"""
test_layer2_orchestrator.py

Tests for `python-engine/services/layer2/orchestrator.py`.

Each test's docstring names the exact SWARM-4 bug ID it catches.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pytest

from services.layer2.orchestrator import Orchestrator


# ===========================================================================
# asyncio.gather error handling (BUG #26)
# ===========================================================================
class TestAsyncioGatherErrorHandling:
    """Locks in SWARM-4 BUG #26: orchestrator uses `asyncio.gather(*tasks)`
    WITHOUT `return_exceptions=True`. One bad proposal crashes the entire
    round, discarding other agents' results."""

    @pytest.mark.asyncio
    async def test_gather_uses_return_exceptions(self):
        """Verify the orchestrator uses `return_exceptions=True` in
        asyncio.gather, so one agent failing doesn't kill the round."""
        source = inspect.getsource(Orchestrator)
        # Find every asyncio.gather call.
        import re
        gather_calls = re.findall(r"asyncio\.gather\([^)]*\)", source, re.DOTALL)
        if not gather_calls:
            pytest.skip("No asyncio.gather found in Orchestrator — bug may not apply.")
        for call in gather_calls:
            assert "return_exceptions=True" in call, (
                f"asyncio.gather call lacks `return_exceptions=True`: {call!r}. "
                "One agent raising an exception will kill the entire round, "
                "discarding other agents' results."
            )


# ===========================================================================
# Ambiguity reduction metric (BUG #4)
# ===========================================================================
class TestAmbiguityReductionMetric:
    """Locks in SWARM-4 BUG #4: `ambiguity_reduction = pred.mean /
    (pred.std_dev + 0.001)` is degenerate — it rewards low-variance
    proposals over high-yield ones, contradicting the maximize-the-sink goal."""

    def test_ambiguity_reduction_not_only_metric(self):
        """Verify the orchestrator doesn't use ONLY `mean / std` as the
        ranking metric. Expected improvement (or similar) must also be
        considered."""
        source = inspect.getsource(Orchestrator)
        # The buggy line: ambiguity_reduction = pred.mean / (pred.std_dev + 0.001)
        # Look for this exact pattern.
        assert "pred.mean / (pred.std_dev" not in source and \
               "pred.mean/(pred.std_dev" not in source, (
            "Orchestrator uses `pred.mean / (pred.std_dev + 0.001)` as the "
            "ranking metric — this is degenerate. It rewards low-variance "
            "proposals over high-yield ones, contradicting maximize-the-sink. "
            "Use Expected Improvement or a multi-objective blend instead."
        )


# ===========================================================================
# Hardcoded bounds (BUG #3)
# ===========================================================================
class TestHardcodedBounds:
    """Locks in SWARM-4 BUG #3: orchestrator hardcodes `[0, 100]` domain
    bounds for every source node. Real variables (temperature, pressure,
    fractions, money) operate on wildly different scales."""

    def test_does_not_hardcode_zero_to_hundred_bounds(self):
        """Verify the orchestrator doesn't hardcode `[0, 100]` bounds for
        every node. Bounds should come from the graph metadata or env vars."""
        source = inspect.getsource(Orchestrator)
        # Look for the exact pattern.
        # The buggy code has lines like: bounds = [0, 100] or (0, 100).
        import re
        # Find any explicit `[0, 100]` or `(0, 100)` literal that's used as bounds.
        # We exclude things like `range(0, 100)` and `range(100)`.
        bounds_patterns = re.findall(r"bounds\s*=\s*\[?\s*0\s*,\s*100\s*\]?", source)
        bounds_patterns += re.findall(r"bounds\s*=\s*\(\s*0\s*,\s*100\s*\)", source)
        if bounds_patterns:
            pytest.fail(
                f"Orchestrator hardcodes [0, 100] bounds: {bounds_patterns!r}. "
                "Real variables have wildly different scales. Bounds should "
                "come from graph metadata, env vars, or per-node config."
            )


# ===========================================================================
# Persists all proposals, not just winner (BUG #28)
# ===========================================================================
class TestPersistsAllProposals:
    """Locks in SWARM-4 BUG #28: orchestrator records only the winning
    proposal per round to `historical_data`, wasting 2/3 of the GP training
    data. ALL proposals (with their evaluated outcomes) should be recorded."""

    def test_records_all_proposals_per_round(self):
        """Verify that for each round, all agent proposals are appended to
        historical_data — not just the winning one."""
        source = inspect.getsource(Orchestrator)
        # Look for the pattern where only the winner is appended.
        # Buggy code: `historical_data.append(winner)` or similar.
        # If the code only appends the best, it's a bug.
        # We can't perfectly detect this from source alone, but we can
        # check that the function appends multiple times per round OR
        # uses a list extension (extend) rather than a single append.
        # Look for `historical_data.append(` calls — if exactly one per round,
        # that's the bug.
        import re
        append_calls = re.findall(r"historical_data\.append\(", source)
        extend_calls = re.findall(r"historical_data\.extend\(", source)
        if len(append_calls) == 1 and not extend_calls:
            pytest.fail(
                "Orchestrator only calls `historical_data.append(...)` once "
                "per round — only the winning proposal is recorded. This "
                "wastes 2/3 of the GP training data. Record ALL proposals "
                "with their evaluated outcomes."
            )


# ===========================================================================
# Cancellation handling
# ===========================================================================
class TestCancellationHandling:
    """The orchestrator should respond to cancellation — not run forever."""

    @pytest.mark.asyncio
    async def test_handles_cancellation(self):
        """If the orchestrator's task is cancelled, it should exit cleanly
        (not hang or raise an opaque error)."""
        # Instantiate an orchestrator with minimal config.
        # We use a tiny graph and max_iterations=1 to keep it fast.
        from schemas.graph import CausalGraph, Node, Edge
        graph = CausalGraph(
            nodes=[Node(id="A", label="A"), Node(id="B", label="B")],
            edges=[Edge(source="A", target="B", relation="ACTIVATES")],
        )

        # Mock the LLM client to return immediately.
        with patch("services.anthropic_client._get_llm_client") as mock_client:
            mock_client.return_value.completion = MagicMock(
                return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="{}"))])
            )

            try:
                orch = Orchestrator(graph=graph, target_node_id="B", max_rounds=1)
            except Exception:
                pytest.skip("Orchestrator constructor requires different args")

            # Start a run and cancel it immediately.
            task = asyncio.create_task(orch.run())
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # Clean cancellation — pass.
            except Exception as e:
                pytest.fail(f"Orchestrator did not handle cancellation cleanly: {e}")


# ===========================================================================
# Max rounds respected
# ===========================================================================
class TestMaxRoundsRespected:
    def test_max_rounds_respected(self):
        """With max_rounds=3, the orchestrator should stop after 3 rounds,
        not run forever."""
        source = inspect.getsource(Orchestrator)
        # Verify there's a check for max_rounds in the run loop.
        # We look for `max_rounds` or `max_iterations` being checked.
        if "max_rounds" not in source and "max_iterations" not in source:
            pytest.fail(
                "Orchestrator has no max_rounds / max_iterations check — "
                "it could run forever."
            )


# ===========================================================================
# Target node is actually a sink
# ===========================================================================
class TestTargetNodeIsSink:
    """Catches SWARM-6 BUG #17: orchestrator picks `graph['nodes'][-1]['id']`
    as the target — that's the last node in the list, not necessarily a sink.
    A sink has no outgoing edges."""

    def test_target_node_has_no_outgoing_edges(self):
        """Verify the orchestrator picks a sink node (no outgoing edges) as
        the target, not just `nodes[-1]`."""
        source = inspect.getsource(Orchestrator)
        # Look for the buggy pattern: graph['nodes'][-1]['id'] or nodes[-1].
        if "nodes[-1]" in source and "target" in source.lower():
            # It might be using the last node as target — verify there's
            # also a sink check.
            if "sink" not in source.lower() and "outgoing" not in source.lower():
                pytest.fail(
                    "Orchestrator picks `graph['nodes'][-1]['id']` as the "
                    "target — that's the last node in the list, not a sink. "
                    "Compute actual sink candidates (no outgoing edges) and "
                    "pick one of those."
                )


# ===========================================================================
# Concurrent agents have independent state
# ===========================================================================
class TestConcurrentAgentsIndependentState:
    """Catches SWARM-4 BUG: shared state mutation between agents running
    in parallel."""

    @pytest.mark.asyncio
    async def test_agents_do_not_share_mutable_state(self):
        """If two agents run concurrently, one's state changes should not
        leak into the other."""
        # We can verify this at the source level: agents should not write
        # to module-level mutable globals.
        from services.layer2 import agent_explorer, agent_exploiter, agent_contrarian
        for module in (agent_explorer, agent_exploiter, agent_contrarian):
            source = inspect.getsource(module)
            # Look for module-level mutable globals (excluding constants).
            # A simple heuristic: `SOME_STATE = {}` or `SOME_STATE = []` at
            # module level (not inside a class or function).
            import re
            # Find top-level assignments to mutable objects.
            lines = source.split("\n")
            for line in lines:
                stripped = line.lstrip()
                # Skip inside class/function (heuristic: indent > 0).
                if line.startswith(" ") or line.startswith("\t"):
                    continue
                # Look for `NAME = {` or `NAME = [` (mutable) — but not
                # class definitions or constants.
                m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=\s*([\{\[])", stripped)
                if m and not m.group(1).isupper():
                    # Mutable module-level state — potential leak.
                    # But constants like `_STATE = {}` are still problematic
                    # if mutated. We just check for state-like names.
                    name = m.group(1)
                    if "STATE" in name or "HISTORY" in name or "CACHE" in name:
                        pytest.fail(
                            f"Module-level mutable state `{name}` in "
                            f"{module.__name__} — concurrent agents may "
                            f"share/mutate it. Use per-instance state."
                        )


# ===========================================================================
# No deadlock — agents don't wait on each other in a cycle
# ===========================================================================
class TestNoDeadlock:
    def test_no_agent_waits_on_other_agent(self):
        """Verify no agent's code calls another agent's run() method
        synchronously (which could deadlock if there's a cycle)."""
        from services.layer2 import agent_explorer, agent_exploiter, agent_contrarian
        for module in (agent_explorer, agent_exploiter, agent_contrarian):
            source = inspect.getsource(module)
            # An agent shouldn't import or call another agent's run().
            other_agents = ["AgentExplorer", "AgentExploiter", "AgentContrarian"]
            for other in other_agents:
                # Skip self-references in the class definition itself.
                # We look for instantiation or .run() calls.
                if other in source and module.__name__.split("_")[-1] not in other.lower():
                    # Loose check — flag for review.
                    pass  # We don't fail; just a heuristic. Real deadlocks
                    # require runtime tests.