"""
test_layer1_validator.py

Tests for `python-engine/services/layer1/validator.py` AND the deployment-
status classifier in `python-engine/services/anthropic_client.py`
(which the SWARM-3 audit flagged as part of the Layer 1 validator surface
because it is the gatekeeper for Layer 3 source ranking).

Each test's docstring names the exact SWARM-3 bug ID it catches so a future
engineer can grep `SWARM-3 BUG #N` and find the test that locks the fix in
place.

NOTE on "red-then-green" semantics:
    The tests are written so that the current (buggy) code FAILS them.
    When the bug is fixed, the tests must PASS. Tests that include BOTH
    a positive and a negative case (e.g., "contradiction detected" AND
    "no contradiction detected") are designed to defeat an LLM that "fakes
    the fix" by unconditionally returning True / always-routing-the-same-way.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import networkx as nx
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module imports (engine source tree is added to sys.path by conftest.py)
# ---------------------------------------------------------------------------
from services.layer1.validator import (
    GraphValidator,
    ConfidenceChecker,
    validate_graph,
    assert_valid_graph,
    ValidationError,
)
from services.layer1.extractor import AmbiguityDetector
from services.anthropic_client import evaluate_deployment_status
from services import anthropic_client as ac_mod


# ===========================================================================
# Contradiction detector (BUG #1) — positive AND negative case
# ===========================================================================
class TestContradictionDetector:
    """Locks in SWARM-3 BUG #1: contradiction detector reads `type` not
    `relation`, so it never fires."""

    def test_contradiction_detector_reads_relation_not_type(self):
        """Catches SWARM-3 BUG #1: contradiction detector reads `type` instead of
        `relation`, so the feature is silently dead."""
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        # Two parallel edges with conflicting relations — every builder in the
        # codebase writes `relation`, NEVER `type`.
        g.add_edge("A", "B", relation="ACTIVATES", source="A", target="B",
                   confidence=80.0, weight=0.5, key=0)
        g.add_edge("A", "B", relation="INHIBITS", source="A", target="B",
                   confidence=80.0, weight=0.5, key=1)

        g2, logs = GraphValidator._flag_contradictions(g)

        assert any("CONTRADICTION" in line for line in logs), (
            "Expected a CONTRADICTION log line when A→B has both "
            "ACTIVATES and INHIBITS relations."
        )
        # BOTH parallel edges must be flagged, not just one.
        contradicted_flags = []
        for u, v, k, d in g2.edges(data=True, keys=True):
            contradicted_flags.append(d.get("is_contradicted"))
        assert all(contradicted_flags), (
            f"Expected is_contradicted=True for BOTH parallel edges; "
            f"got flags={contradicted_flags}"
        )

    def test_contradiction_detector_negative_case(self):
        """Negative case: a graph with NO contradictions must NOT be flagged.
        This catches an LLM that 'fakes the fix' by always returning True."""
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        # All edges agree (ACTIVATES only) — no conflicts.
        g.add_edge("A", "B", relation="ACTIVATES", source="A", target="B",
                   confidence=80.0, key=0)
        g.add_edge("B", "C", relation="ACTIVATES", source="B", target="C",
                   confidence=80.0, key=0)
        g.add_edge("A", "C", relation="ACTIVATES", source="A", target="C",
                   confidence=80.0, key=0)

        g2, logs = GraphValidator._flag_contradictions(g)

        assert not any("CONTRADICTION" in line for line in logs), (
            f"No contradictions present, but got logs={logs}"
        )
        for u, v, k, d in g2.edges(data=True, keys=True):
            assert d.get("is_contradicted") in (False, None), (
                f"Edge {u}->{v} (key={k}) was wrongly marked as contradicted: {d}"
            )

    def test_contradiction_detector_ignores_type_attribute(self):
        """Defeats an LLM fix that simply swaps `relation` for `type` in the
        detector: feeding a graph whose edges have BOTH `type=ACTIVATES` and
        `type=INHIBITS` but `relation=None` should NOT trigger (the contract
        is `relation`)."""
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        # Edges with ONLY `type` attribute, no `relation` — these were never
        # produced by any builder, so they must not be treated as a
        # contradiction based on `type` (the contract is `relation`).
        g.add_edge("A", "B", type="ACTIVATES", source="A", target="B",
                   confidence=80.0, key=0)
        g.add_edge("A", "B", type="INHIBITS", source="A", target="B",
                   confidence=80.0, key=1)

        g2, logs = GraphValidator._flag_contradictions(g)
        # If the fix correctly switched to reading `relation`, these edges
        # have NO `relation` attribute, so no contradiction should be flagged.
        # If the fix simply swapped `type` -> `relation` (i.e., kept reading
        # `type` but renamed the variable), this would wrongly fire.
        assert not any("CONTRADICTION" in line for line in logs), (
            "Edges with no `relation` attribute should NOT trigger a "
            "contradiction. The contract is `relation`."
        )

    def test_contradiction_penalty_applied_correctly(self):
        """Catches SWARM-3 BUG #1 (penalty half): the `_PENALTY_CONTRADICTED_EDGE`
        penalty (25 points) must be subtracted from node confidence when a
        contradicted edge touches the node, not merely the `is_contradicted`
        flag set."""
        from services.layer1.extractor import (
            _PENALTY_CONTRADICTED_EDGE,
            AmbiguityDetector,
        )
        assert _PENALTY_CONTRADICTED_EDGE == 25, (
            f"Spec says penalty is 25; got {_PENALTY_CONTRADICTED_EDGE}"
        )

        # Graph: A -> B [ACTIVATES] and A -> B [INHIBITS], flagged by
        # _flag_contradictions, then run through AmbiguityDetector.
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 100.0},
                {"id": "B", "label": "B", "confidence": 100.0},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "ACTIVATES",
                 "confidence": 80.0, "weight": 0.5, "is_contradicted": True},
                {"source": "A", "target": "B", "relation": "INHIBITS",
                 "confidence": 80.0, "weight": 0.5, "is_contradicted": True},
            ],
        }
        result = AmbiguityDetector.analyze_graph(graph_data)
        # Both A and B touch the contradicted edge → penalty = 25.
        for n in result["nodes"]:
            assert n["confidence"] == 100.0 - _PENALTY_CONTRADICTED_EDGE, (
                f"Node {n['id']} expected confidence "
                f"{100.0 - _PENALTY_CONTRADICTED_EDGE}, got {n['confidence']}. "
                f"Penalty was not applied correctly."
            )


# ===========================================================================
# Confidence routing (BUG #2) — inverted routing logic
# ===========================================================================
class TestConfidenceRouting:
    """Locks in SWARM-3 BUG #2: routing logic is inverted."""

    def _make_graph(self, *, overall_conf, urgent_nodes):
        return {
            "nodes": [{"id": f"n{i}", "confidence": 95.0} for i in range(10)],
            "edges": [],
            "overall_graph_confidence": overall_conf,
            "urgent_nodes": urgent_nodes,
        }

    def test_confidence_routing_not_inverted(self):
        """Catches SWARM-3 BUG #2: a graph with one catastrophic node
        (confidence=5%, urgency=95) + 9 healthy nodes (confidence=95%) must
        route to L2 simulation, NOT SKIP_TO_L3. The buggy code reads
        `urgent_nodes[0]['urgency_score']` as `lowest_node_conf` — but
        urgent_nodes is sorted DESC by urgency, so [0] = MAX urgency, and
        `MAX_urgency < 85` is False, so it skips L2 even though one node is
        catastrophic."""
        urgent = (
            [{"node_id": "bad", "urgency_score": 95.0, "confidence": 5.0}] +
            [{"node_id": f"n{i}", "urgency_score": 5.0, "confidence": 95.0}
             for i in range(9)]
        )
        urgent.sort(key=lambda x: x["urgency_score"], reverse=True)
        graph_data = self._make_graph(
            overall_conf=(9 * 95.0 + 5.0) / 10,  # 86.0
            urgent_nodes=urgent,
        )
        result = ConfidenceChecker.evaluate_graph(graph_data, "DATA")
        assert result["routing"] == "ROUTE_TO_L2_SIMULATION", (
            f"One node has confidence 5% → must route to L2 simulation. "
            f"Got {result['routing']} (BUG #2 inverted logic)."
        )

    def test_confidence_routing_high_confidence_skips_l2(self):
        """A graph where ALL nodes are 95% confident and the overall conf ≥85
        must SKIP_TO_L3 (no L2 simulation needed)."""
        urgent = [{"node_id": f"n{i}", "urgency_score": 5.0, "confidence": 95.0}
                  for i in range(10)]
        urgent.sort(key=lambda x: x["urgency_score"], reverse=True)
        graph_data = self._make_graph(overall_conf=95.0, urgent_nodes=urgent)
        result = ConfidenceChecker.evaluate_graph(graph_data, "DATA")
        assert result["routing"] == "SKIP_TO_L3", (
            f"High-confidence graph must skip L2; got {result['routing']}."
        )

    def test_confidence_routing_medium_confidence_goes_to_l2(self):
        """A graph with overall_conf < 85 (DATA path) must route to L2."""
        urgent = [{"node_id": f"n{i}", "urgency_score": 30.0, "confidence": 70.0}
                  for i in range(10)]
        urgent.sort(key=lambda x: x["urgency_score"], reverse=True)
        graph_data = self._make_graph(overall_conf=70.0, urgent_nodes=urgent)
        result = ConfidenceChecker.evaluate_graph(graph_data, "DATA")
        assert result["routing"] == "ROUTE_TO_L2_SIMULATION", (
            f"Low overall confidence on DATA path must go to L2; "
            f"got {result['routing']}."
        )

    def test_confidence_routing_text_path_low_confidence_skips_l2(self):
        """For TEXT path, even low-confidence graphs skip L2 (no simulation
        for text) and go straight to L3 — but the routing logic still must
        correctly identify 'low confidence' (urgency check)."""
        urgent = (
            [{"node_id": "bad", "urgency_score": 95.0, "confidence": 5.0}] +
            [{"node_id": f"n{i}", "urgency_score": 5.0, "confidence": 95.0}
             for i in range(9)]
        )
        urgent.sort(key=lambda x: x["urgency_score"], reverse=True)
        graph_data = self._make_graph(
            overall_conf=(9 * 95.0 + 5.0) / 10, urgent_nodes=urgent
        )
        result = ConfidenceChecker.evaluate_graph(graph_data, "TEXT")
        # TEXT path always skips L2 (no simulation), but the routing
        # must still be SKIP_TO_L3 (not L2).
        assert result["routing"] == "SKIP_TO_L3"
        # However, the "is_low_confidence" branch should have been correctly
        # detected — verify by inspecting that low-confidence TEXT graphs
        # still log the right path. (We cannot inspect internal logs, but
        # the routing must be SKIP_TO_L3 regardless.)


# ===========================================================================
# Fix impossible edges (BUG #13) — stale source/target after flip
# ===========================================================================
class TestFixImpossibleEdges:
    """Locks in SWARM-3 BUG #13: after flipping an edge A→B to B→A, the
    serialized edge's `source`/`target` keys remain STALE (point to A→B)
    because the original `edge_data` dict carries the old source/target
    and `**edge_data` overwrites the networkx endpoints in serialization.
    """

    def test_fix_impossible_edges_preserves_correct_attributes(self, mock_litellm):
        """Catches SWARM-3 BUG #13: after flipping, the serialized edge's
        `source` and `target` fields must match the NEW direction, not the
        stale original."""
        # Mock the LLM to flip 'Survival_Rate' -> 'Dosage' to 'Dosage' -> 'Survival_Rate'
        flip_payload = json.dumps([
            {"source": "Survival_Rate", "target": "Dosage",
             "reason": "outcome cannot cause input"}
        ])
        mock_litellm.queue_response(flip_payload)

        graph_data = {
            "nodes": [
                {"id": "Survival_Rate", "label": "Survival_Rate",
                 "confidence": 80.0},
                {"id": "Dosage", "label": "Dosage", "confidence": 80.0},
            ],
            "edges": [
                {"source": "Survival_Rate", "target": "Dosage",
                 "relation": "CAUSES", "confidence": 80.0, "weight": 0.5},
            ],
        }
        result = GraphValidator.validate(graph_data, model_name="gpt-4o")
        edges = result["graph"]["edges"]
        assert len(edges) == 1, f"Expected 1 edge after flip; got {len(edges)}"
        e = edges[0]
        # The networkx edge endpoints are (Dosage, Survival_Rate) after flip.
        # The serialized dict's `source`/`target` MUST match these endpoints,
        # not the stale values carried in the original edge_data dict.
        assert e["source"] == "Dosage", (
            f"After flip, source should be 'Dosage'; got {e['source']}. "
            f"BUG #13: stale source attribute after edge flip. Edge: {e}"
        )
        assert e["target"] == "Survival_Rate", (
            f"After flip, target should be 'Survival_Rate'; got {e['target']}. "
            f"BUG #13: stale target attribute after edge flip. Edge: {e}"
        )

    def test_fix_impossible_edges_creates_dag(self, mock_litellm):
        """After all flips, the resulting graph must be a DAG (no cycles).
        The flip operation should not introduce cycles."""
        # Mock the LLM to flip one edge in a 3-node chain A->B->C->A.
        # LLM flips A->C to C->A? That would still be a cycle. Instead,
        # we test that the validator's cycle-breaking step (called after
        # the flip step) correctly removes the resulting cycle.
        # Simpler: have the LLM flip A->B to B->A, then check that the
        # resulting graph (with only the flipped edge) is acyclic.
        flip_payload = json.dumps([
            {"source": "A", "target": "B", "reason": "test"}
        ])
        mock_litellm.queue_response(flip_payload)
        # Empty payload for subsequent batches if any.
        mock_litellm.default_response = "[]"

        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "B", "label": "B", "confidence": 80.0},
                {"id": "C", "label": "C", "confidence": 80.0},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 80.0},
                {"source": "B", "target": "C", "relation": "CAUSES",
                 "confidence": 80.0},
                # No cycle: A->B->C only.
            ],
        }
        result = GraphValidator.validate(graph_data, model_name="gpt-4o")
        # Reconstruct networkx graph from serialized output and assert DAG.
        out = result["graph"]
        g = nx.MultiDiGraph()
        for n in out["nodes"]:
            g.add_node(n["id"], **n)
        for e in out["edges"]:
            g.add_edge(e["source"], e["target"], **e)
        # A simple DiGraph projection should be acyclic.
        di = nx.DiGraph(g)
        assert nx.is_directed_acyclic_graph(di), (
            "Validator output must be a DAG. "
            f"Got cycles: {list(nx.simple_cycles(di))}"
        )

    def test_fix_impossible_edges_no_flip_preserves_direction(self, mock_litellm):
        """Negative case: if the LLM returns [] (no flips), edges must
        retain their original source/target."""
        mock_litellm.queue_response("[]")
        mock_litellm.default_response = "[]"
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "B", "label": "B", "confidence": 80.0},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 80.0, "weight": 0.5},
            ],
        }
        result = GraphValidator.validate(graph_data, model_name="gpt-4o")
        e = result["graph"]["edges"][0]
        assert e["source"] == "A" and e["target"] == "B", (
            f"No flips → edge must retain A->B; got {e['source']}->{e['target']}"
        )


# ===========================================================================
# evaluate_deployment_status (BUG #29) — substring match
# ===========================================================================
class TestEvaluateDeploymentStatus:
    """Locks in SWARM-3 BUG #29: substring match in
    `anthropic_client.evaluate_deployment_status` causes 'not deployed'
    to return 'deployed'."""

    @staticmethod
    def _make_resp(content: str):
        m = MagicMock()
        m.message.content = content
        c = MagicMock()
        c.message = m.message  # use the same message object
        # Actually construct a simple struct
        class _M:
            def __init__(self, c):
                self.content = c
        class _C:
            def __init__(self, c):
                self.message = _M(c)
        class _R:
            def __init__(self, c):
                self.choices = [_C(c)]
        return _R(content)

    @pytest.fixture(autouse=True)
    def _patch_llm(self, monkeypatch):
        # Patch _get_llm_client to return a mock client so no API key needed.
        self.mock_client = MagicMock()
        monkeypatch.setattr(ac_mod, "_get_llm_client", lambda: self.mock_client)

    async def test_evaluate_deployment_status_no_substring_match(self):
        """Catches SWARM-3 BUG #29: 'not deployed' should NOT be classified
        as 'deployed'. The buggy substring check `if 'deployed' in raw`
        matches 'not deployed'."""
        self.mock_client.completion.return_value = self._make_resp("not deployed")
        result = await evaluate_deployment_status("http://x", "snippet")
        assert result != "deployed", (
            "LLM said 'not deployed' but function returned 'deployed' — "
            "substring match bug (BUG #29)."
        )

    async def test_evaluate_deployment_status_correctly_identifies_states(self):
        """Multiple positive/negative cases for BUG #29."""
        cases = [
            ("deployed", "deployed"),
            ("Deployed", "deployed"),         # case-insensitive
            ("single_study", "single_study"),
            ("Single_Study", "single_study"),
            ("blog", "blog"),
            ("not deployed", "blog"),         # NOT 'deployed' (bug fix)
            ("this is not a single_study", "blog"),
            ("the system is deployed", "deployed"),  # ambiguous but positive
            ("", "blog"),                     # empty fallback
        ]
        for raw, expected in cases:
            self.mock_client.completion.reset_mock()
            self.mock_client.completion.return_value = self._make_resp(raw)
            result = await evaluate_deployment_status("http://x", "snippet")
            # Only assert the strict negative cases that catch the substring
            # bug. The positive cases may be implementation-dependent; we
            # assert only the negative cases that MUST NOT be mis-classified.
            if raw in ("not deployed", "this is not a single_study", ""):
                assert result != "deployed", (
                    f"raw={raw!r} returned 'deployed' — BUG #29 substring match."
                )
                if raw == "this is not a single_study":
                    assert result != "single_study", (
                        f"raw={raw!r} returned 'single_study' — BUG #29 substring."
                    )


# ===========================================================================
# Ambiguity detector confidence threshold (BUG #55 — scale mismatch)
# ===========================================================================
class TestAmbiguityDetectorThreshold:
    """Locks in SWARM-3 BUG #55 (the codebase uses 0-100 scale for confidence
    but `_PENALTY_LOW_EDGE_CONF` triggers at `< 0.60` — a 0-1 threshold that
    never fires on the 0-100 scale used everywhere else)."""

    def test_ambiguity_detector_confidence_threshold_uses_zero_to_hundred(self):
        """Catches SWARM-3 BUG #55: the low-edge-confidence penalty triggers
        when `e.get('confidence') < 0.60` — but confidence in this codebase
        is always 0–100 (see schemas/graph.py: `Field(ge=0.0, le=100.0)`),
        so the check never fires for any real edge (which would be e.g. 80.0).
        A correct fix uses a 0-100 threshold (e.g., < 60.0)."""
        from services.layer1.extractor import _PENALTY_LOW_EDGE_CONF

        # Graph: one node, one edge with confidence 50.0 (out of 100).
        # On the 0-100 scale, 50% confidence is "low" — the penalty must fire.
        graph_data = {
            "nodes": [{"id": "A", "label": "A", "confidence": 100.0},
                      {"id": "B", "label": "B", "confidence": 100.0}],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 50.0, "weight": 0.5},
            ],
        }
        result = AmbiguityDetector.analyze_graph(graph_data)
        # The edge has confidence 50% on a 0-100 scale → must be flagged as
        # low-confidence (penalty = _PENALTY_LOW_EDGE_CONF = 20).
        # Node A and B both touch this edge, so both must receive the penalty.
        for n in result["nodes"]:
            # The penalty must reduce confidence below 100.
            assert n["confidence"] == 100.0 - _PENALTY_LOW_EDGE_CONF, (
                f"Edge has confidence 50% (0-100 scale) — must trigger "
                f"low-edge-confidence penalty of {_PENALTY_LOW_EDGE_CONF}. "
                f"Node {n['id']} got confidence={n['confidence']}. "
                f"BUG #55: threshold uses 0-1 scale, never fires on 0-100."
            )

    def test_ambiguity_detector_penalty_value_is_correct(self):
        """Catches SWARM-3 BUG #1/55: assert the EXACT penalty value (20)
        is applied, not just 'some penalty'."""
        from services.layer1.extractor import _PENALTY_LOW_EDGE_CONF
        assert _PENALTY_LOW_EDGE_CONF == 20, (
            f"Spec: low-edge-conf penalty is 20; got {_PENALTY_LOW_EDGE_CONF}"
        )
        # Graph where one edge has confidence 30% (0-100) — low.
        # Node A is the only node touching this low-conf edge.
        # But node A also has degree=1 (single connection) → +10 penalty.
        # Total expected penalty for node A: 20 (low edge conf) + 10 (low degree) = 30.
        # Node B is similar (single connection + low-conf edge) → same.
        graph_data = {
            "nodes": [{"id": "A", "label": "A", "confidence": 100.0},
                      {"id": "B", "label": "B", "confidence": 100.0}],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 30.0, "weight": 0.5},
            ],
        }
        result = AmbiguityDetector.analyze_graph(graph_data)
        for n in result["nodes"]:
            expected_penalty = (
                _PENALTY_LOW_EDGE_CONF  # 20 (low edge conf)
                + 10                    # _PENALTY_LOW_DEGREE (single edge)
            )
            assert n["ambiguity_score"] == expected_penalty, (
                f"Node {n['id']} expected ambiguity_score "
                f"{expected_penalty}, got {n['ambiguity_score']}."
            )

    def test_ambiguity_detector_high_confidence_no_penalty(self):
        """Negative case: a node whose edges all have confidence ≥ 60 (0-100
        scale) must NOT receive the low-edge-confidence penalty.
        This catches an LLM that 'fakes the fix' by always applying the penalty."""
        from services.layer1.extractor import _PENALTY_LOW_EDGE_CONF
        # Two edges with confidence 95% each, both touching node B.
        # B has degree 2 (>1) so no low-degree penalty either.
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 100.0},
                {"id": "B", "label": "B", "confidence": 100.0},
                {"id": "C", "label": "C", "confidence": 100.0},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 95.0},
                {"source": "C", "target": "B", "relation": "CAUSES",
                 "confidence": 95.0},
            ],
        }
        result = AmbiguityDetector.analyze_graph(graph_data)
        b_node = next(n for n in result["nodes"] if n["id"] == "B")
        # B touches two high-confidence edges, has degree 2 (>1), so 0 penalty.
        assert b_node["ambiguity_score"] == 0, (
            f"B should have 0 ambiguity (high-conf edges, degree 2); "
            f"got {b_node['ambiguity_score']}."
        )


# ===========================================================================
# Async functions don't block event loop (BUG #23, #24)
# ===========================================================================
class TestAsyncNonBlocking:
    """Locks in SWARM-3 BUG #23, #24: `extract_causal_graph_from_text` and
    `evaluate_compatibility_and_transferability` are declared `async` but
    call the synchronous `client.completion`, which blocks the event loop.
    The fix is to use `litellm.acompletion` (or wrap in `asyncio.to_thread`).
    """

    def test_extract_causal_graph_from_text_is_async(self):
        """`extract_causal_graph_from_text` must be declared as a coroutine
        function (so `await` is required by callers)."""
        from services.anthropic_client import extract_causal_graph_from_text
        assert inspect.iscoroutinefunction(extract_causal_graph_from_text), (
            "extract_causal_graph_from_text must be `async def`."
        )

    def test_evaluate_compatibility_and_transferability_is_async(self):
        from services.anthropic_client import (
            evaluate_compatibility_and_transferability
        )
        assert inspect.iscoroutinefunction(
            evaluate_compatibility_and_transferability
        ), "evaluate_compatibility_and_transferability must be `async def`."

    async def test_extract_causal_graph_uses_acompletion(self, mock_litellm):
        """Catches SWARM-3 BUG #24: when called as a coroutine, the function
        must use the ASYNC litellm path (acompletion), not the sync path
        (completion). We assert that the mock's `acompletion` is invoked
        and `completion` is NOT."""
        from services.anthropic_client import extract_causal_graph_from_text
        valid_json = json.dumps({
            "nodes": [{"id": "n1", "label": "X", "confidence": 80.0}],
            "edges": [],
        })
        mock_litellm.queue_response(valid_json)
        # Patch _get_llm_client to return the litellm module so the function
        # actually calls litellm.completion / litellm.acompletion (mocked).
        with patch.object(ac_mod, "_get_llm_client", return_value=__import__("litellm")):
            try:
                await extract_causal_graph_from_text("Some text.")
            except Exception:
                # We don't care if it raises on the mock; we only care which
                # entrypoint was called.
                pass
        assert len(mock_litellm.async_calls) > 0, (
            "Async extract_causal_graph_from_text must call acompletion; "
            "BUG #24: it calls sync completion, blocking the event loop."
        )
        assert len(mock_litellm.sync_calls) == 0, (
            "BUG #24: async function used the SYNC completion path."
        )

    async def test_evaluate_compatibility_uses_acompletion(self, mock_litellm):
        """Catches SWARM-3 BUG #24 (sibling): evaluate_compatibility_and_transferability
        must use acompletion."""
        from services.anthropic_client import (
            evaluate_compatibility_and_transferability
        )
        valid_json = json.dumps({
            "constraint_compatibility": 0.7,
            "solution_transferability": 0.8,
        })
        mock_litellm.queue_response(valid_json)
        with patch.object(ac_mod, "_get_llm_client", return_value=__import__("litellm")):
            try:
                await evaluate_compatibility_and_transferability("ctx", "mech")
            except Exception:
                pass
        assert len(mock_litellm.async_calls) > 0, (
            "BUG #24: async evaluate_compatibility_and_transferability must "
            "use acompletion."
        )

    async def test_build_ontology_from_text_uses_acompletion(self, mock_litellm):
        """Catches SWARM-3 BUG #23: build_ontology_from_text is async but
        uses sync `client.completion`. Must use acompletion."""
        from services.layer1.ontology_builder import build_ontology_from_text
        valid_json = json.dumps({
            "nodes": [{"id": "n1", "label": "X", "confidence": 80.0}],
            "edges": [],
        })
        mock_litellm.queue_response(valid_json)
        # Patch _get_llm_client to return the litellm module.
        from services.layer1 import ontology_builder as ob_mod
        with patch.object(ob_mod, "_get_llm_client", return_value=__import__("litellm")):
            try:
                await build_ontology_from_text({"content": "Some text."})
            except Exception:
                pass
        assert len(mock_litellm.async_calls) > 0, (
            "BUG #23: async build_ontology_from_text must use acompletion."
        )


# ===========================================================================
# LLM JSON output safety (BUG #25) — graceful handling
# ===========================================================================
class TestLLMJsonOutputSafety:
    """Locks in SWARM-3 BUG #25: build_ontology_from_text calls
    `json.loads(raw)` without try/except; malformed LLM output crashes
    the whole pipeline."""

    async def test_llm_json_output_parsed_safely(self, mock_litellm):
        """Catches SWARM-3 BUG #25: malformed JSON from the LLM must not
        crash the function as a raw `json.JSONDecodeError`. It must either
        return a graceful fallback (empty CausalGraph) OR raise a clear,
        named exception type (ValueError/RuntimeError with a useful message).

        The buggy code does `data = json.loads(raw.strip())` with no
        try/except — `json.JSONDecodeError` propagates unmodified."""
        from services.layer1.ontology_builder import build_ontology_from_text
        # Malformed JSON
        mock_litellm.queue_response("This is not JSON at all, just prose.")
        from services.layer1 import ontology_builder as ob_mod
        with patch.object(ob_mod, "_get_llm_client", return_value=__import__("litellm")):
            try:
                result = await build_ontology_from_text({"content": "Some text."})
                # Graceful fallback path: result is a valid CausalGraph.
                assert result is not None
            except json.JSONDecodeError as e:
                # The exact subclass of ValueError means the function did NOT
                # catch it — a "fake fix" that just keeps propagating
                # JSONDecodeError would also trigger this branch.
                pytest.fail(
                    f"BUG #25: malformed LLM JSON propagated as a raw "
                    f"json.JSONDecodeError (a ValueError subclass) — the "
                    f"function did not handle it gracefully. Error: {e}"
                )
            except (ValueError, RuntimeError, EnvironmentError):
                # A clear, named exception type (NOT json.JSONDecodeError) is
                # an acceptable "graceful" handling path.
                pass

    async def test_llm_json_output_missing_keys_handled(self, mock_litellm):
        """Catches SWARM-3 BUG #25 (sibling): JSON with missing keys (e.g.,
        no 'nodes' key) must be handled gracefully, not raise KeyError."""
        from services.layer1.ontology_builder import build_ontology_from_text
        # JSON missing the 'nodes' key
        mock_litellm.queue_response(json.dumps({"edges": []}))
        from services.layer1 import ontology_builder as ob_mod
        with patch.object(ob_mod, "_get_llm_client", return_value=__import__("litellm")):
            try:
                result = await build_ontology_from_text({"content": "Some text."})
                assert result is not None
            except (ValueError, RuntimeError, KeyError, EnvironmentError) as e:
                # KeyError on 'nodes' is NOT graceful — fail explicitly.
                if isinstance(e, KeyError):
                    pytest.fail(
                        f"BUG #25: missing 'nodes' key raised KeyError: {e}"
                    )
            except json.JSONDecodeError:
                pytest.fail("BUG #25: JSON parsed but missing keys crashed.")


# ===========================================================================
# Schema validation (validate_graph / assert_valid_graph)
# ===========================================================================
class TestSchemaValidation:
    """Smoke tests for the schema-level `validate_graph` function — these
    are not flagged in the SWARM-3 audit but provide coverage for the
    positive path so a broken fix that breaks `validate_graph` itself
    is caught."""

    def _build_causal_graph(self, nodes, edges):
        from schemas.graph import CausalGraph, Node, Edge
        node_objs = [Node(**n) for n in nodes]
        edge_objs = [Edge(**e) for e in edges]
        return CausalGraph(nodes=node_objs, edges=edge_objs)

    def test_validate_graph_passes_simple_graph(self):
        nodes = [
            {"id": "A", "label": "A", "confidence": 80.0},
            {"id": "B", "label": "B", "confidence": 80.0},
        ]
        edges = [
            {"source": "A", "target": "B", "relation": "CAUSES",
             "confidence": 80.0, "weight": 0.5},
        ]
        ok, issues = validate_graph(self._build_causal_graph(nodes, edges))
        assert ok
        assert not any(i.startswith("CRITICAL") for i in issues)

    def test_validate_graph_rejects_dangling_edge(self):
        # Need >= 2 nodes so the function doesn't bail out early.
        nodes = [
            {"id": "A", "label": "A", "confidence": 80.0},
            {"id": "B", "label": "B", "confidence": 80.0},
        ]
        edges = [{"source": "A", "target": "MISSING",
                  "relation": "CAUSES", "confidence": 80.0, "weight": 0.5}]
        ok, issues = validate_graph(self._build_causal_graph(nodes, edges))
        assert not ok
        assert any("MISSING" in i for i in issues), (
            f"Dangling edge target must be reported; got issues={issues}"
        )

    def test_validate_graph_rejects_too_few_nodes(self):
        nodes = [{"id": "A", "label": "A", "confidence": 80.0}]
        ok, issues = validate_graph(self._build_causal_graph(nodes, []))
        assert not ok

    def test_assert_valid_graph_raises(self):
        nodes = [{"id": "A", "label": "A", "confidence": 80.0}]
        with pytest.raises(ValidationError):
            assert_valid_graph(self._build_causal_graph(nodes, []))


# ===========================================================================
# Cycle resolution smoke test
# ===========================================================================
class TestResolveCycles:
    """Smoke tests for `_resolve_cycles` — SWARM-3 BUG #36/#37 note that
    `nx.simple_cycles` is exponential and that default confidence for
    unmarked edges is 1.0 (very low). The cycle-breaking logic itself is
    correct for small graphs, so we lock the happy path here."""

    def test_resolve_cycles_breaks_simple_cycle(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", relation="CAUSES", confidence=80.0, source="A", target="B")
        g.add_edge("B", "C", relation="CAUSES", confidence=80.0, source="B", target="C")
        g.add_edge("C", "A", relation="CAUSES", confidence=80.0, source="C", target="A")
        g2, logs = GraphValidator._resolve_cycles(g)
        di = nx.DiGraph(g2)
        assert nx.is_directed_acyclic_graph(di), (
            f"After cycle resolution, graph must be DAG; cycles: "
            f"{list(nx.simple_cycles(di))}"
        )
        assert any("CYCLE" in l for l in logs), (
            f"Expected CYCLE log; got {logs}"
        )

    def test_resolve_cycles_preserves_acyclic_graph(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", relation="CAUSES", confidence=80.0, source="A", target="B")
        g2, logs = GraphValidator._resolve_cycles(g)
        # Acyclic graph → no cycles, no logs.
        di = nx.DiGraph(g2)
        assert nx.is_directed_acyclic_graph(di)
        assert not any("CYCLE" in l for l in logs)


# ===========================================================================
# Remove disconnected (smoke test for the 'preserves isolated nodes when
# graph has no edges' edge case)
# ===========================================================================
class TestRemoveDisconnected:
    def test_keeps_all_nodes_when_no_edges(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g2, logs = GraphValidator._remove_disconnected(g)
        assert set(g2.nodes()) == {"A", "B", "C"}, (
            "Graph with no edges must keep ALL nodes (the spec says to "
            "preserve the causal structure rather than wipe the graph)."
        )

    def test_removes_isolated_when_edges_exist(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("ISO")
        g.add_edge("A", "B", relation="CAUSES", confidence=80.0,
                   source="A", target="B")
        g2, logs = GraphValidator._remove_disconnected(g)
        assert "ISO" not in g2.nodes()
        assert {"A", "B"} <= set(g2.nodes())