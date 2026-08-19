import asyncio
import inspect
import logging
from typing import List, Optional, Set

from schemas.graph import CausalGraph, Node, Edge
from schemas.layer3 import StructuralQuery
from services.anthropic_client import generate_domain_blind_query

logger = logging.getLogger(__name__)


def sanitize_causal_graph(graph: Optional[CausalGraph]) -> CausalGraph:
    """
    Validates and sanitizes a CausalGraph to handle graph boundary edge cases:
    - Removes duplicate nodes (retaining unique node IDs)
    - Fixes missing or invalid confidence scores (clamped to [0.0, 100.0])
    - Filters out dangling edges where source or target does not exist in nodes
    - Handles isolated nodes and cyclic references safely
    """
    if not graph or not getattr(graph, "nodes", None):
        return CausalGraph(nodes=[], edges=[])

    valid_nodes: List[Node] = []
    seen_ids: Set[str] = set()

    for idx, node in enumerate(graph.nodes):
        if not node:
            continue
        raw_id = getattr(node, "id", None)
        node_id = str(raw_id).strip() if raw_id is not None else f"node_{idx}"
        if not node_id:
            node_id = f"node_{idx}"
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)

        raw_label = getattr(node, "label", None)
        label = str(raw_label).strip() if raw_label is not None else node_id
        if not label:
            label = node_id

        raw_conf = getattr(node, "confidence", 80.0)
        try:
            conf = max(0.0, min(100.0, float(raw_conf))) if raw_conf is not None else 80.0
        except (ValueError, TypeError):
            conf = 80.0

        is_unknown = bool(getattr(node, "is_unknown", False))
        domain = getattr(node, "domain", None)
        description = getattr(node, "description", None)

        valid_nodes.append(Node(
            id=node_id,
            label=label,
            domain=domain,
            confidence=conf,
            is_unknown=is_unknown,
            description=description,
        ))

    valid_edges: List[Edge] = []
    if getattr(graph, "edges", None):
        seen_edges: Set[tuple] = set()
        for edge in graph.edges:
            if not edge:
                continue
            src = str(getattr(edge, "source", "") or "").strip()
            tgt = str(getattr(edge, "target", "") or "").strip()
            # Guard against dangling edges (missing source or target nodes)
            if not src or not tgt or src not in seen_ids or tgt not in seen_ids:
                continue

            rel = str(getattr(edge, "relation", "CAUSES") or "CAUSES").strip().upper()
            raw_econf = getattr(edge, "confidence", 80.0)
            try:
                econf = max(0.0, min(100.0, float(raw_econf))) if raw_econf is not None else 80.0
            except (ValueError, TypeError):
                econf = 80.0

            raw_weight = getattr(edge, "weight", 1.0)
            try:
                eweight = float(raw_weight) if raw_weight is not None else 1.0
            except (ValueError, TypeError):
                eweight = 1.0

            edge_key = (src, tgt, rel)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            valid_edges.append(Edge(
                source=src,
                target=tgt,
                relation=rel,
                confidence=econf,
                weight=eweight,
            ))

    return CausalGraph(nodes=valid_nodes, edges=valid_edges)


def extract_unknown_nodes(graph: Optional[CausalGraph], confidence_threshold: float = 80.0) -> List[Node]:
    """
    Extracts all unknown nodes, based on explicit flag or low confidence.
    Gracefully handles empty/null graphs, isolated nodes, cycles, and missing target nodes.
    """
    if not graph or not getattr(graph, "nodes", None):
        return []

    sanitized = sanitize_causal_graph(graph)
    unknowns: List[Node] = []
    for node in sanitized.nodes:
        if node.is_unknown or node.confidence < confidence_threshold:
            unknowns.append(node)
    return unknowns


def rank_nodes(nodes: List[Node], graph: Optional[CausalGraph] = None) -> List[Node]:
    """
    Ranks nodes by importance:
    1. Lowest confidence first (highest uncertainty)
    2. Higher connectivity / degree (central bottlenecks prioritized over isolated nodes)
    Gracefully handles empty lists, isolated nodes, and cycles.
    """
    if not nodes:
        return []

    node_degree: dict = {}
    if graph and getattr(graph, "edges", None):
        for edge in graph.edges:
            if edge and getattr(edge, "source", None) and getattr(edge, "target", None):
                node_degree[edge.source] = node_degree.get(edge.source, 0) + 1
                node_degree[edge.target] = node_degree.get(edge.target, 0) + 1

    def _sort_key(n: Node):
        conf = getattr(n, "confidence", 0.0)
        if conf is None:
            conf = 0.0
        deg = node_degree.get(getattr(n, "id", ""), 0)
        return (conf, -deg)

    return sorted(nodes, key=_sort_key)


def run_step_10(graph: Optional[CausalGraph], confidence_threshold: float = 80.0) -> List[StructuralQuery]:
    """
    Main orchestrator for Layer 3 Step 10:
    1. Sanitize graph (handling boundary edge cases: missing targets, cycles, isolated nodes)
    2. Extract unknown nodes based on confidence threshold or explicit is_unknown flag
    3. Rank them by uncertainty and structural importance
    4. Convert each into a domain-blind structural query
    5. Handle exceptions per node so a single LLM failure does not kill Step 10 (Fixes BUG #36)
    """
    if not graph or not getattr(graph, "nodes", None):
        logger.warning("[unknown_extractor] Empty or null graph provided to run_step_10.")
        return []

    sanitized_graph = sanitize_causal_graph(graph)
    unknown_nodes = extract_unknown_nodes(sanitized_graph, confidence_threshold)
    if not unknown_nodes:
        logger.info("[unknown_extractor] No unknown nodes found in graph.")
        return []

    ranked_nodes = rank_nodes(unknown_nodes, sanitized_graph)

    queries: List[StructuralQuery] = []
    for node in ranked_nodes:
        try:
            res = generate_domain_blind_query(node, sanitized_graph)
            if inspect.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        structural_desc = pool.submit(asyncio.run, res).result()
                else:
                    structural_desc = asyncio.run(res)
            else:
                structural_desc = res
        except Exception as exc:
            logger.error(
                f"[unknown_extractor] LLM call failed for node '{node.id}': {exc}. Skipping."
            )
            continue

        if not structural_desc or not isinstance(structural_desc, str) or not structural_desc.strip():
            logger.warning(
                f"[unknown_extractor] Empty or invalid structural description generated for node '{node.id}'. Skipping."
            )
            continue

        queries.append(StructuralQuery(
            original_node_id=str(node.id),
            original_confidence=node.confidence,
            structural_description=structural_desc.strip(),
        ))

    return queries
