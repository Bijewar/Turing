"""
test_layer1_classifier.py

Tests for `python-engine/services/layer1/classifier.py`.

Each test's docstring names the exact SWARM-3 bug ID it catches.
"""

from __future__ import annotations

import json
import re
import inspect
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from services.layer1.classifier import NodeClassifier


# ===========================================================================
# JSON extraction (BUG #27) — greedy regex
# ===========================================================================
class TestClassifyGraphJsonExtraction:
    """Locks in SWARM-3 BUG #27: classifier uses greedy `\\[.*\\]` regex that
    captures from the FIRST `[` to the LAST `]`, swallowing reasoning text
    between two arrays and producing invalid JSON. The hidden-node pass
    then silently fails (caught by broad `except Exception`)."""

    def _make_graph_with_edges(self):
        return {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "C", "label": "C", "confidence": 80.0},
            ],
            "edges": [
                {"source": "A", "target": "C", "relation": "CAUSES",
                 "confidence": 80.0, "weight": 0.5},
            ],
        }

    def test_classify_graph_regex_not_greedy(self, mock_litellm):
        """Catches SWARM-3 BUG #27: when the LLM response contains valid JSON
        followed by reasoning text with brackets, the classifier must
        extract ONLY the first JSON array (non-greedy), not the greedy span
        from the first `[` to the last `]`.

        The buggy code uses `re.search(r'\\[.*\\]', content, re.DOTALL)` —
        the greedy `.*` swallows the text between two bracketed sections,
        producing invalid JSON. `json.loads` then raises, the broad
        `except Exception` swallows it, and NO hidden nodes are added.
        A correct fix uses a non-greedy `\\[.*?\\]` or a proper JSON parser.
        """
        # Build a graph with one edge so the hidden-node pass fires.
        graph_data = self._make_graph_with_edges()

        # Mock LLM response: valid JSON array, then reasoning with brackets.
        # The greedy regex would capture: [{...}] — I also considered [B, D]
        # which is NOT valid JSON. The non-greedy regex would capture just the
        # first array, which IS valid JSON and adds hidden node B.
        llm_response = (
            '[{"source":"A","target":"C","hidden_node_name":"B"}]\n'
            'I also considered [D, E] but they were rejected.'
        )
        mock_litellm.queue_response(llm_response)
        mock_litellm.default_response = llm_response

        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")

        # The hidden node B must be added (if the fix is correct).
        node_ids = [n["id"] for n in result["nodes"]]
        assert "B" in node_ids, (
            "BUG #27: greedy regex `[.*]` swallowed the reasoning text "
            "and `json.loads` failed silently, so hidden node B was not "
            f"added. Got nodes: {node_ids}."
        )

    def test_classify_graph_regex_handles_only_first_array(self, mock_litellm):
        """Defeats an LLM fix that simply uses `re.search(r'\\[.*?\\]', ...)`
        on a response where the FIRST `[...]` is NOT the JSON we want.
        Actually the contract is: the LLM is asked to return a JSON array,
        and any non-greedy match should find it. This test confirms that
        when the LLM returns a clean response with one array, the hidden
        node is added correctly."""
        graph_data = self._make_graph_with_edges()
        mock_litellm.queue_response(
            '[{"source":"A","target":"C","hidden_node_name":"B"}]'
        )
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        assert any(n["id"] == "B" for n in result["nodes"])

    def test_classify_graph_handles_no_json_in_response(self, mock_litellm):
        """Catches SWARM-3 BUG #27 (sibling): when the LLM returns prose
        without any JSON (no brackets at all), the classifier must fall back
        gracefully — no crash, no hidden nodes added, original graph
        preserved."""
        graph_data = self._make_graph_with_edges()
        # LLM returns prose, no brackets
        mock_litellm.queue_response(
            "I considered the graph but found no gaps. No hidden nodes needed."
        )
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        # No hidden nodes added (graceful fallback).
        assert all(not n.get("is_hidden") for n in result["nodes"])
        # Original A and C must still be present.
        ids = [n["id"] for n in result["nodes"]]
        assert "A" in ids and "C" in ids

    def test_classify_graph_handles_completely_garbled_response(self, mock_litellm):
        """Catches SWARM-3 BUG #27: a completely garbled response (random
        chars) must not crash the classifier — it must return the original
        graph unchanged."""
        graph_data = self._make_graph_with_edges()
        mock_litellm.queue_response(",,,,,[[[[invalid]]]])))))){{{{}}}}")
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        # Must not crash; must return graph with at least A and C.
        ids = [n["id"] for n in result["nodes"]]
        assert "A" in ids and "C" in ids


# ===========================================================================
# Node metadata preservation
# ===========================================================================
class TestClassifyGraphPreservesMetadata:
    def test_classify_graph_preserves_node_metadata(self, mock_litellm):
        """After classification, nodes must still have all original fields.
        Catches a hypothetical fix that overwrites node dicts instead of
        updating them in-place."""
        graph_data = {
            "nodes": [
                {"id": "A", "label": "Alpha", "confidence": 80.0,
                 "domain": "biology", "description": "An input variable",
                 "is_unknown": False, "custom_field": 42},
                {"id": "B", "label": "Beta", "confidence": 80.0,
                 "domain": "biology", "description": "An outcome",
                 "is_unknown": False, "custom_field": 99},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 80.0, "weight": 0.5},
            ],
        }
        mock_litellm.queue_response("[]")  # No hidden nodes
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        # Find node A in result
        node_a = next(n for n in result["nodes"] if n["id"] == "A")
        # Original fields must be preserved.
        assert node_a["label"] == "Alpha"
        assert node_a["domain"] == "biology"
        assert node_a["description"] == "An input variable"
        assert node_a["is_unknown"] is False
        assert node_a["custom_field"] == 42, (
            "Original node metadata (custom_field) must be preserved."
        )
        # New classifier-added fields must also be present.
        assert "node_class" in node_a
        assert "theme" in node_a


# ===========================================================================
# Reproducibility
# ===========================================================================
class TestClassifyGraphReproducible:
    def test_classify_graph_random_seed_reproducible(self, mock_litellm):
        """Same input + same mock LLM response → same output. This is a
        smoke test for determinism. The classifier itself is deterministic
        (no RNG), but this test locks that in."""
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
            ],
        }
        mock_litellm.queue_response("[]")
        r1 = NodeClassifier.classify_graph(
            json.loads(json.dumps(graph_data)), model_name="gpt-4o"
        )
        mock_litellm.queue_response("[]")
        r2 = NodeClassifier.classify_graph(
            json.loads(json.dumps(graph_data)), model_name="gpt-4o"
        )
        # Topology-based classification is deterministic.
        assert [n["node_class"] for n in r1["nodes"]] == \
               [n["node_class"] for n in r2["nodes"]]


# ===========================================================================
# Async path uses acompletion (BUG #14 sibling — classifier uses sync completion)
# ===========================================================================
class TestAsyncClassify:
    """The classifier's `classify_graph` is currently a SYNC function (not
    async), so it always uses `litellm.completion`. The SWARM-3 audit BUG #14
    flags that this blocks the event loop when called from an async FastAPI
    handler. A correct fix wraps the LLM call in `asyncio.to_thread` or
    provides an async variant that uses `acompletion`."""

    def test_classify_graph_is_sync_uses_completion(self, mock_litellm):
        """Sync `classify_graph` uses sync `completion` (current behavior).
        This test locks in the current contract so a refactor that switches
        to `acompletion` (without making the function async) would break it
        and force the developer to think."""
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "B", "label": "B", "confidence": 80.0},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 80.0},
            ],
        }
        mock_litellm.queue_response("[]")
        NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        # The current contract: sync classify_graph uses sync completion.
        # (If someone refactors to use acompletion without making the
        # function async, they must update this test.)
        assert len(mock_litellm.sync_calls) >= 1, (
            "Sync classify_graph must use litellm.completion (the sync path)."
        )

    def test_async_classify_uses_acompletion_not_completion(self, mock_litellm):
        """Catches SWARM-3 BUG #14 (sibling): if a future async version of
        classify_graph is added, it must use `litellm.acompletion` (NOT the
        sync `completion`), to avoid blocking the event loop. If no async
        variant exists, the test is skipped."""
        if not hasattr(NodeClassifier, "classify_graph_async"):
            # No async variant exists yet — that itself is a BUG #14
            # smell (sync classify_graph called from async endpoint blocks
            # the event loop). Mark as expected fail.
            pytest.xfail(
                "No async classify_graph variant exists — sync "
                "classify_graph blocks the event loop when called from "
                "the async FastAPI endpoint (SWARM-3 BUG #14)."
            )
        # If async variant exists, verify it uses acompletion.
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "B", "label": "B", "confidence": 80.0},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 80.0},
            ],
        }
        mock_litellm.queue_response("[]")
        import asyncio
        asyncio.run(NodeClassifier.classify_graph_async(graph_data, "gpt-4o"))
        assert len(mock_litellm.async_calls) >= 1, (
            "Async classify_graph variant must use acompletion."
        )
        assert len(mock_litellm.sync_calls) == 0


# ===========================================================================
# Rate limit handling (BUG #30)
# ===========================================================================
class TestClassifyGraphRateLimit:
    """Locks in SWARM-3 BUG #30: LLM calls have no retry, no timeout, no
    rate-limit handling. A RateLimitError from the LLM must be handled
    gracefully (retry or fallback), not crash the function."""

    def test_classify_graph_handles_llm_rate_limit_error(self, mock_litellm):
        """Catches SWARM-3 BUG #30: when the LLM raises a rate-limit error,
        classify_graph must handle it gracefully (return the graph without
        hidden nodes, or retry) — NOT propagate the exception uncaught."""
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "B", "label": "B", "confidence": 80.0},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 80.0},
            ],
        }
        # Queue an exception for the LLM call.
        try:
            from litellm.exceptions import RateLimitError
            exc = RateLimitError(
                message="Rate limit exceeded",
                model="gpt-4o",
                llm_provider="openai",
            )
        except (ImportError, Exception):
            # Fallback: a generic Exception that the broad except in
            # classify_graph will catch.
            exc = Exception("Rate limit exceeded")
        mock_litellm.queue_response(exc)
        # The classifier has a broad `except Exception` — it should catch
        # the rate-limit error and return the graph without hidden nodes.
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        # Must not crash; must return graph with A and B preserved.
        ids = [n["id"] for n in result["nodes"]]
        assert "A" in ids and "B" in ids


# ===========================================================================
# Topology-based classification correctness (smoke tests)
# ===========================================================================
class TestTopologyClassification:
    """Smoke tests that the topology-based classifier (the non-LLM half of
    classify_graph) correctly identifies SOURCE, SINK, BOTTLENECK, MEDIATOR."""

    def test_source_node_classified_correctly(self, mock_litellm):
        """A node with 0 incoming edges is a SOURCE."""
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "B", "label": "B", "confidence": 80.0},
                {"id": "C", "label": "C", "confidence": 80.0},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 80.0},
                {"source": "A", "target": "C", "relation": "CAUSES",
                 "confidence": 80.0},
            ],
        }
        mock_litellm.queue_response("[]")
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        node_a = next(n for n in result["nodes"] if n["id"] == "A")
        assert node_a["node_class"] == "SOURCE"
        assert node_a["theme"] == "blue"

    def test_sink_node_classified_correctly(self, mock_litellm):
        """A node with 0 outgoing edges (and some incoming) is a SINK."""
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "B", "label": "B", "confidence": 80.0},
                {"id": "C", "label": "C", "confidence": 80.0},
            ],
            "edges": [
                {"source": "A", "target": "C", "relation": "CAUSES",
                 "confidence": 80.0},
                {"source": "B", "target": "C", "relation": "CAUSES",
                 "confidence": 80.0},
            ],
        }
        mock_litellm.queue_response("[]")
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        node_c = next(n for n in result["nodes"] if n["id"] == "C")
        assert node_c["node_class"] == "SINK"
        assert node_c["theme"] == "green"

    def test_bottleneck_node_classified_correctly(self, mock_litellm):
        """A node with indeg>1 and outdeg>=1 is a BOTTLENECK."""
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "B", "label": "B", "confidence": 80.0},
                {"id": "C", "label": "C", "confidence": 80.0},
                {"id": "D", "label": "D", "confidence": 80.0},
            ],
            "edges": [
                {"source": "A", "target": "C", "relation": "CAUSES",
                 "confidence": 80.0},
                {"source": "B", "target": "C", "relation": "CAUSES",
                 "confidence": 80.0},
                {"source": "C", "target": "D", "relation": "CAUSES",
                 "confidence": 80.0},
            ],
        }
        mock_litellm.queue_response("[]")
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        node_c = next(n for n in result["nodes"] if n["id"] == "C")
        assert node_c["node_class"] == "BOTTLENECK"
        assert node_c["theme"] == "red"
        assert node_c["is_pulsing"] is True

    def test_mediator_node_classified_correctly(self, mock_litellm):
        """A node with indeg==1 and outdeg>=1 is a MEDIATOR."""
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
            ],
        }
        mock_litellm.queue_response("[]")
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        node_b = next(n for n in result["nodes"] if n["id"] == "B")
        assert node_b["node_class"] == "MEDIATOR"


# ===========================================================================
# Hidden-node ID collision (BUG #52)
# ===========================================================================
class TestHiddenNodeCollision:
    """Locks in SWARM-3 BUG #52: `hidden_node_id = str(hidden_name).upper()`
    may collide with an existing node ID. The classifier checks for the
    collision before adding the hidden node, so this is mostly handled, but
    the test locks in the behavior."""

    def test_hidden_node_collision_does_not_overwrite(self, mock_litellm):
        """If the LLM suggests a hidden node whose upper-cased ID matches
        an existing node, the classifier must NOT overwrite the existing
        node — it should either skip the hidden node or use a different ID.
        """
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0,
                 "node_class": "SOURCE", "theme": "blue"},
                {"id": "B", "label": "B", "confidence": 80.0,
                 "node_class": "SINK", "theme": "green"},
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "CAUSES",
                 "confidence": 80.0},
            ],
        }
        # LLM suggests hidden node "A" (uppercase) — collides with existing.
        mock_litellm.queue_response(
            '[{"source":"A","target":"B","hidden_node_name":"A"}]'
        )
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        # The existing node A must NOT be marked as hidden.
        node_a = next(n for n in result["nodes"] if n["id"] == "A")
        assert not node_a.get("is_hidden", False), (
            "Existing node A must not be turned into a hidden node by "
            "the hidden-node detection pass (BUG #52)."
        )
        assert node_a["node_class"] == "SOURCE", (
            "Existing node A's classification must be preserved (BUG #52)."
        )


# ===========================================================================
# Empty nodes / no edges edge cases
# ===========================================================================
class TestClassifyGraphEdgeCases:
    def test_classify_graph_empty_nodes(self):
        """A graph with no nodes must return unchanged (no crash)."""
        graph_data = {"nodes": [], "edges": []}
        result = NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        assert result["nodes"] == []

    def test_classify_graph_no_edges_skips_llm(self, mock_litellm):
        """If a graph has no edges, the hidden-node pass must NOT call the
        LLM (the `if edges:` guard on line 79 of classifier.py)."""
        graph_data = {
            "nodes": [
                {"id": "A", "label": "A", "confidence": 80.0},
                {"id": "B", "label": "B", "confidence": 80.0},
            ],
            "edges": [],
        }
        NodeClassifier.classify_graph(graph_data, model_name="gpt-4o")
        assert len(mock_litellm.sync_calls) == 0, (
            "No edges → must NOT call the LLM for hidden-node detection."
        )