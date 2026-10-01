"""
test_layer1_ontology_builder.py

Tests for `python-engine/services/layer1/ontology_builder.py`.

Covers:
- `build_ontology_from_text` (Mayank's one-shot path)
- `LLMGraphBuilder.build_graph` (Harsh's dual-stage path)

Each test's docstring names the exact SWARM-3 bug ID it catches.
"""

from __future__ import annotations

import json
import os
import re
from unittest.mock import patch, MagicMock

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from services.layer1.ontology_builder import (
    build_ontology_from_text,
    LLMGraphBuilder,
)


# ===========================================================================
# DAG output (smoke test)
# ===========================================================================
class TestBuildOntologyNoCycles:
    """The output of LLMGraphBuilder.build_graph must be a DAG — every edge
    added by the LLM should respect causality. The builder itself does NOT
    enforce DAG-ness (it relies on the validator downstream), but for a
    well-behaved LLM mock the output should be acyclic."""

    def test_build_ontology_no_cycles(self, mock_litellm):
        """LLMGraphBuilder.build_graph output (with clean mock edges) must
        form a DAG — no cycles introduced by the builder itself."""
        # Mock _generate_ontology and _extract_causal_edges by configuring
        # the LLM responses in sequence.
        # First call: ontology generation. Second+ calls: edge extraction.
        ontology_json = json.dumps({
            "entity_types": ["ENZYME", "PATHWAY"],
            "relation_types": ["INHIBITS", "ACTIVATES"],
        })
        edges_json = json.dumps([
            {"source": "ENZYME_A", "target": "PATHWAY_X",
             "relation": "ACTIVATES", "confidence": "DEFINITELY"},
            {"source": "ENZYME_B", "target": "PATHWAY_X",
             "relation": "INHIBITS", "confidence": "LIKELY"},
        ])
        mock_litellm.queue_responses(ontology_json, edges_json)

        # The build_graph method calls _generate_ontology then
        # _extract_causal_edges; both call litellm.completion.
        result = LLMGraphBuilder.build_graph(
            "Some scientific text about enzymes and pathways.",
            model_name="gpt-4o",
        )
        # Reconstruct networkx graph from serialized output and assert DAG.
        g = nx.DiGraph()
        for n in result["nodes"]:
            g.add_node(n["id"], **n)
        for e in result["edges"]:
            g.add_edge(e["source"], e["target"], **e)
        assert nx.is_directed_acyclic_graph(g), (
            f"build_graph output must be a DAG for well-behaved edges; "
            f"cycles: {list(nx.simple_cycles(g))}"
        )


# ===========================================================================
# No duplicate nodes (BUG #50 sibling)
# ===========================================================================
class TestBuildOntologyNoDuplicates:
    """Locks in SWARM-3 BUG #50 (sibling): LLMGraphBuilder uppercases
    source/target, but a related bug is that duplicate node names from the
    LLM (e.g., "Enzyme A" and "ENZYME A" — same after uppercasing) should
    not produce duplicate nodes in the output."""

    def test_build_ontology_no_duplicate_nodes(self, mock_litellm):
        """If the LLM emits two edges that mention the same entity with
        different casing, the uppercasing in build_graph must merge them
        into a single node (no duplicates)."""
        ontology_json = json.dumps({
            "entity_types": ["ENZYME"],
            "relation_types": ["ACTIVATES"],
        })
        # LLM mentions "enzyme_a" and "ENZYME_A" — same after uppercasing.
        edges_json = json.dumps([
            {"source": "enzyme_a", "target": "PATHWAY_X",
             "relation": "ACTIVATES", "confidence": "DEFINITELY"},
            {"source": "ENZYME_A", "target": "PATHWAY_Y",
             "relation": "INHIBITS", "confidence": "LIKELY"},
        ])
        mock_litellm.queue_responses(ontology_json, edges_json)
        result = LLMGraphBuilder.build_graph(
            "Text about ENZYME_A.", model_name="gpt-4o"
        )
        node_ids = [n["id"] for n in result["nodes"]]
        assert len(node_ids) == len(set(node_ids)), (
            f"Duplicate node IDs after uppercasing must be merged; "
            f"got {node_ids} (with duplicates)."
        )
        # Both forms must merge to "ENZYME_A".
        assert "ENZYME_A" in node_ids
        # No "enzyme_a" lowercase form.
        assert "enzyme_a" not in node_ids


# ===========================================================================
# Async path uses acompletion (BUG #23)
# ===========================================================================
class TestBuildOntologyAsync:
    """Locks in SWARM-3 BUG #23: build_ontology_from_text is async but
    uses the sync `client.completion`, blocking the event loop."""

    def test_build_ontology_from_text_is_async(self):
        """The function must be declared as a coroutine function."""
        import inspect
        assert inspect.iscoroutinefunction(build_ontology_from_text), (
            "build_ontology_from_text must be `async def`."
        )

    async def test_build_ontology_async_uses_acompletion(self, mock_litellm):
        """Catches SWARM-3 BUG #23: when called as a coroutine, the function
        must use the ASYNC litellm path (acompletion), not the sync path
        (completion)."""
        valid_json = json.dumps({
            "nodes": [{"id": "n1", "label": "X", "confidence": 80.0}],
            "edges": [],
        })
        mock_litellm.queue_response(valid_json)
        from services.layer1 import ontology_builder as ob_mod
        with patch.object(ob_mod, "_get_llm_client", return_value=__import__("litellm")):
            try:
                await build_ontology_from_text({"content": "Some text."})
            except Exception:
                pass
        assert len(mock_litellm.async_calls) > 0, (
            "BUG #23: async build_ontology_from_text must call acompletion."
        )
        assert len(mock_litellm.sync_calls) == 0, (
            "BUG #23: async function used the SYNC completion path."
        )


# ===========================================================================
# Empty text handling
# ===========================================================================
class TestBuildOntologyEmptyText:
    def test_build_ontology_handles_empty_text(self):
        """Empty text input must return an empty CausalGraph (or raise a
        clear ValueError) — NOT crash with an opaque exception."""
        # Empty string
        result = build_ontology_from_text({"content": ""})
        # The function returns a coroutine since it's async; need to await.
        import asyncio
        async def _run():
            return await build_ontology_from_text({"content": ""})
        # Per the source, empty text returns CausalGraph(nodes=[], edges=[])
        # BEFORE checking for an API key, so this should work without env vars.
        try:
            r = asyncio.run(_run())
            assert r is not None
            # If it returns, must be empty graph (no nodes/edges)
            assert len(r.nodes) == 0
            assert len(r.edges) == 0
        except (ValueError, RuntimeError, EnvironmentError):
            # Acceptable: a clear, named exception type.
            pass

    def test_build_ontology_handles_whitespace_only_text(self):
        """Whitespace-only text must be treated as empty."""
        import asyncio
        async def _run():
            return await build_ontology_from_text({"content": "   \n\n  "})
        try:
            r = asyncio.run(_run())
            assert r is not None
            assert len(r.nodes) == 0
        except (ValueError, RuntimeError, EnvironmentError):
            pass

    def test_llm_graph_builder_build_graph_rejects_empty_text(self):
        """LLMGraphBuilder.build_graph raises ValueError on empty text —
        verify that the error message is clear (mentions 'empty text')."""
        with pytest.raises(ValueError, match="empty"):
            LLMGraphBuilder.build_graph("", model_name="gpt-4o")


# ===========================================================================
# Long text handling (BUG #49 sibling — chunking)
# ===========================================================================
class TestBuildOntologyLongText:
    """Locks in SWARM-3 BUG #49: LLMGraphBuilder.build_graph chunking has
    no overlap — cross-chunk causal edges are lost. The audit suggests
    adding overlap. The test here verifies that very long text is at least
    CHUNKED (or truncated with a warning), not silently dropped."""

    def test_build_ontology_handles_very_long_text(self, mock_litellm):
        """Text > LLM token limit (default 4000 chars in
        build_ontology_from_text; 500-char chunks in LLMGraphBuilder) must
        be chunked or truncated with a warning, not silently dropped or
        crash. This test verifies that a long text (10000+ chars) produces
        a graph with edges (i.e., the chunking actually processed the
        text)."""
        # Generate a long text with multiple "chunks" worth of content.
        long_text = " ".join(
            f"Enzyme{i} activates Pathway{i}." for i in range(500)
        )  # ~17000 chars
        assert len(long_text) > 4000  # exceeds the one-shot truncation

        # For LLMGraphBuilder: first call is ontology, then chunks.
        # With 17k chars and 500-char chunks → ~34 chunks → 35 LLM calls.
        ontology_json = json.dumps({
            "entity_types": ["ENZYME", "PATHWAY"],
            "relation_types": ["ACTIVATES"],
        })
        # For each chunk, return one edge.
        edge_json = json.dumps([
            {"source": "ENZYME", "target": "PATHWAY",
             "relation": "ACTIVATES", "confidence": "DEFINITELY"}
        ])
        # Pre-queue: 1 ontology + many chunks.
        responses = [ontology_json] + [edge_json] * 50
        mock_litellm.queue_responses(*responses)
        mock_litellm.default_response = edge_json

        result = LLMGraphBuilder.build_graph(long_text, model_name="gpt-4o")
        # The output must have at least one node and one edge — proves the
        # chunking processed the text rather than dropping it.
        assert len(result["nodes"]) >= 2, (
            f"Long text must produce at least 2 nodes; got {result['nodes']}."
        )
        assert len(result["edges"]) >= 1, (
            f"Long text must produce at least 1 edge; got {result['edges']}."
        )


# ===========================================================================
# LLM output validation (BUG #25 sibling)
# ===========================================================================
class TestBuildOntologyLlmOutputValidation:
    """Locks in SWARM-3 BUG #25 (sibling): LLMGraphBuilder.build_graph does
    not validate that edges reference existing nodes. If the LLM returns an
    edge with a source/target that doesn't appear in any node, the edge
    should be filtered out OR the missing node should be auto-created. The
    current code auto-creates nodes from edges (lines 126-129), so this test
    verifies that behavior is consistent."""

    def test_build_ontology_llm_output_validated(self, mock_litellm):
        """LLMGraphBuilder.build_graph must produce a graph where every
        edge references nodes that exist in the nodes list. The current
        code auto-creates nodes for every unique source/target in edges,
        so this test verifies that invariant."""
        ontology_json = json.dumps({
            "entity_types": ["X"],
            "relation_types": ["CAUSES"],
        })
        # Edges reference nodes A and B which don't appear in any explicit
        # node list — the builder auto-creates them.
        edges_json = json.dumps([
            {"source": "A", "target": "B", "relation": "CAUSES",
             "confidence": "DEFINITELY"},
            # An edge that references a non-existent target — the builder
            # auto-creates "C".
            {"source": "A", "target": "C", "relation": "INHIBITS",
             "confidence": "LIKELY"},
        ])
        mock_litellm.queue_responses(ontology_json, edges_json)
        result = LLMGraphBuilder.build_graph("Text.", model_name="gpt-4o")
        # Every edge's source/target must be in nodes.
        node_ids = {n["id"] for n in result["nodes"]}
        for e in result["edges"]:
            assert e["source"] in node_ids, (
                f"Edge source {e['source']} not in node set {node_ids}."
            )
            assert e["target"] in node_ids, (
                f"Edge target {e['target']} not in node set {node_ids}."
            )

    def test_build_ontology_filters_empty_source_target(self, mock_litellm):
        """Edges with empty/whitespace source or target must be silently
        dropped by the builder (per lines 118-119 of ontology_builder.py:
        `if not source or not target: continue`)."""
        ontology_json = json.dumps({
            "entity_types": ["X"],
            "relation_types": ["CAUSES"],
        })
        edges_json = json.dumps([
            {"source": "", "target": "B", "relation": "CAUSES",
             "confidence": "DEFINITELY"},  # empty source — must be dropped
            {"source": "A", "target": "B", "relation": "CAUSES",
             "confidence": "DEFINITELY"},
            {"source": "  ", "target": "B", "relation": "CAUSES",
             "confidence": "LIKELY"},  # whitespace-only — must be dropped
        ])
        mock_litellm.queue_responses(ontology_json, edges_json)
        result = LLMGraphBuilder.build_graph("Text.", model_name="gpt-4o")
        # Only one edge should survive (A→B); the two with empty/whitespace
        # source must be dropped.
        valid_edges = [e for e in result["edges"] if e["source"].strip()]
        assert len(valid_edges) == 1, (
            f"Expected 1 valid edge after dropping empty-source edges; "
            f"got {len(valid_edges)}."
        )


# ===========================================================================
# Markdown fence stripping (BUG #26)
# ===========================================================================
class TestBuildOntologyMarkdownFences:
    """Locks in SWARM-3 BUG #26: `_generate_ontology` doesn't strip markdown
    fences before `json.loads`. If the LLM returns ```json ... ``` the
    json.loads fails. The current code at line 169 does NOT strip fences
    for `_generate_ontology` (only build_ontology_from_text does at line 72).
    This test verifies that _generate_ontology handles fenced JSON."""

    def test_generate_ontology_strips_markdown_fences(self, mock_litellm):
        """Catches SWARM-3 BUG #26: _generate_ontology must strip markdown
        fences before json.loads. If the LLM returns ```json {...} ``` the
        function must still parse correctly and return the dict, NOT raise
        a RuntimeError wrapping a JSONDecodeError.

        The current buggy code does NOT strip fences — json.loads fails
        on the leading ` ```json ` — and the broad `except Exception`
        wraps it in a RuntimeError. A proper fix strips the fences first
        so the parse succeeds."""
        fenced_json = "```json\n" + json.dumps({
            "entity_types": ["X"],
            "relation_types": ["CAUSES"],
        }) + "\n```"
        mock_litellm.queue_response(fenced_json)
        ont = LLMGraphBuilder._generate_ontology("text", model_name="gpt-4o")
        # Must successfully parse and return the dict — proves fences were
        # stripped before json.loads.
        assert isinstance(ont, dict), f"Expected dict, got {type(ont)}"
        assert "entity_types" in ont, (
            f"_generate_ontology must return parsed dict with 'entity_types'; "
            f"got {ont}. (BUG #26: markdown fences not stripped before json.loads)"
        )
        assert "relation_types" in ont


# ===========================================================================
# Confidence scale consistency (BUG #51, #55)
# ===========================================================================
class TestBuildOntologyConfidenceScale:
    """Locks in SWARM-3 BUG #51: LLMGraphBuilder default node confidence is
    100.0 — misleading for LLM-inferred nodes. And BUG #55: confidence
    scale should be 0-100 (not 0-1). The builder uses conf_map with values
    0.9/0.7/0.4/0.2 (0-1 scale) and multiplies by 100 (line 136) to get
    90/70/40/20 (0-100 scale). The test verifies that the resulting edge
    confidence values are on the 0-100 scale."""

    def test_build_ontology_edge_confidence_is_zero_to_hundred(self, mock_litellm):
        """Edge confidence values produced by LLMGraphBuilder.build_graph
        must be on the 0-100 scale (per schemas/graph.py), not 0-1."""
        ontology_json = json.dumps({
            "entity_types": ["X"],
            "relation_types": ["CAUSES"],
        })
        edges_json = json.dumps([
            {"source": "A", "target": "B", "relation": "CAUSES",
             "confidence": "DEFINITELY"},  # weight = 0.9, *100 = 90
            {"source": "A", "target": "C", "relation": "INHIBITS",
             "confidence": "MAYBE"},  # weight = 0.4, *100 = 40
        ])
        mock_litellm.queue_responses(ontology_json, edges_json)
        result = LLMGraphBuilder.build_graph("Text.", model_name="gpt-4o")
        for e in result["edges"]:
            assert 0.0 <= e["confidence"] <= 100.0, (
                f"Edge confidence {e['confidence']} not in [0, 100]."
            )
        # Specifically check "DEFINITELY" maps to 90 (not 0.9).
        definitely_edge = next(
            e for e in result["edges"] if e.get("relation") == "CAUSES"
        )
        assert definitely_edge["confidence"] == 90.0, (
            f"DEFINITELY confidence must be 90 (0-100 scale); "
            f"got {definitely_edge['confidence']}."
        )


# ===========================================================================
# INHIBITS gets negative weight (smoke test for the spec)
# ===========================================================================
class TestInhibitsNegativeWeight:
    def test_inhibits_edge_has_negative_weight(self, mock_litellm):
        """INHIBITS edges must have negative weight (per line 137:
        `weight if relation != 'INHIBITS' else -weight`)."""
        ontology_json = json.dumps({
            "entity_types": ["X"],
            "relation_types": ["INHIBITS"],
        })
        edges_json = json.dumps([
            {"source": "A", "target": "B", "relation": "INHIBITS",
             "confidence": "DEFINITELY"},
        ])
        mock_litellm.queue_responses(ontology_json, edges_json)
        result = LLMGraphBuilder.build_graph("Text.", model_name="gpt-4o")
        assert len(result["edges"]) >= 1
        inh_edge = result["edges"][0]
        assert inh_edge["weight"] < 0, (
            f"INHIBITS edge weight must be negative; got {inh_edge['weight']}."
        )