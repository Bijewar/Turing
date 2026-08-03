import asyncio
import networkx as nx

from schemas.graph import CausalGraph
from schemas.layer3 import Step13Request, Step13Response, IsomorphismMatch, MatchType, ExtractedMechanism


def _to_digraph(graph: CausalGraph) -> nx.DiGraph:
    """Converts a CausalGraph schema object into a networkx DiGraph."""
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.id)
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)
    return G


def calculate_structural_similarity(target: CausalGraph, candidate: CausalGraph) -> float:
    """
    Calculates structural similarity using NetworkX's true Graph Edit Distance (GED).
    Replaces arbitrary magic number weights with a mathematically rigorous
    structural comparison.
    """
    target_nx = _to_digraph(target)
    candidate_nx = _to_digraph(candidate)

    t_nodes = target_nx.number_of_nodes()
    c_nodes = candidate_nx.number_of_nodes()

    if t_nodes == 0 and c_nodes == 0:
        return 100.0
    if t_nodes == 0 or c_nodes == 0:
        return 0.0

    def _fallback_ged() -> float:
        if nx.is_isomorphic(target_nx, candidate_nx):
            return 0.0
        in_t = sorted(d for _, d in target_nx.in_degree())
        in_c = sorted(d for _, d in candidate_nx.in_degree())
        out_t = sorted(d for _, d in target_nx.out_degree())
        out_c = sorted(d for _, d in candidate_nx.out_degree())

        max_in = max(len(in_t), len(in_c))
        in_t_pad = [0] * (max_in - len(in_t)) + in_t
        in_c_pad = [0] * (max_in - len(in_c)) + in_c

        max_out = max(len(out_t), len(out_c))
        out_t_pad = [0] * (max_out - len(out_t)) + out_t
        out_c_pad = [0] * (max_out - len(out_c)) + out_c

        diff_in = sum(abs(a - b) for a, b in zip(in_t_pad, in_c_pad))
        diff_out = sum(abs(a - b) for a, b in zip(out_t_pad, out_c_pad))
        diff_nodes = abs(t_nodes - c_nodes)
        diff_edges = abs(target_nx.number_of_edges() - candidate_nx.number_of_edges())

        return float(max(diff_in + diff_out + diff_nodes, diff_edges + diff_nodes, 1.0))

    # Timeout applied to prevent NP-Hard hang on massive graphs
    try:
        ged = nx.graph_edit_distance(target_nx, candidate_nx, timeout=3.0)
        if ged is None:
            ged = _fallback_ged()
    except Exception:
        ged = _fallback_ged()

    # Maximum possible GED is the total sum of all nodes and edges in both graphs
    # (i.e. deleting everything in target, and inserting everything in candidate)
    max_ged = (t_nodes + target_nx.number_of_edges()) + (c_nodes + candidate_nx.number_of_edges())
    
    if max_ged == 0:
        return 100.0

    similarity = max(0.0, 1.0 - (ged / max_ged))
    return round(similarity * 100.0, 2)


async def match_graphs(request: Step13Request) -> Step13Response:
    """
    Step 13: Graph Isomorphism Matcher
    Compares each candidate ExtractedMechanism against the user's target graph
    using structural isomorphism scoring in parallel non-blocking worker threads.
    Thresholds are configurable via request.thresholds.
    """
    thresholds = request.thresholds
    candidates = request.candidates

    if not candidates:
        return Step13Response(matches=[])

    # Run GED similarity calculations in parallel thread pool
    scores = await asyncio.gather(*[
        asyncio.to_thread(calculate_structural_similarity, request.target_graph, candidate.causal_graph)
        for candidate in candidates
    ])

    matches = []
    for candidate, score in zip(candidates, scores):
        if score >= thresholds.perfect:
            match_type = MatchType.PERFECT
        elif score >= thresholds.strong_partial:
            match_type = MatchType.STRONG_PARTIAL
        elif score >= thresholds.weak_partial:
            match_type = MatchType.WEAK_PARTIAL
        else:
            match_type = MatchType.DISCARDED

        matches.append(IsomorphismMatch(
            mechanism=candidate,
            isomorphism_score=score,
            match_type=match_type,
        ))

    return Step13Response(matches=matches)
