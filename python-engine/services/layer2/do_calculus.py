import warnings
from collections import deque
from typing import Dict, List, Set, Optional, Tuple
from schemas.layer2 import GraphEdge, GaussianPrediction, SimulationStepOutput
from services.layer1.gaussian_process import GPEngine


class DoCalculusSimulator:
    """
    Do-Calculus Engine.
    Simulates interventions by cutting incoming edges to the intervened nodes,
    forcing their values, and propagating forward through the causal chain.
    """

    def __init__(self, base_noise: float = 0.02):
        self.gp_engine = GPEngine(base_noise=base_noise)

    def simulate(
        self,
        nodes: List[str],
        edges: List[GraphEdge],
        interventions: Dict[str, float],
    ) -> SimulationStepOutput:
        """
        Run a do-calculus intervention simulation.

        Args:
            nodes:         Full list of node names in the graph.
            edges:         Directed edges (source → target).
            interventions: Dict of {node_name: forced_value} — these nodes have
                           their incoming edges cut and their value fixed.

        Returns:
            SimulationStepOutput with a GaussianPrediction for every reachable node.

        Raises:
            ValueError: if any edge references a node not in `nodes`, or if a cycle is detected.
        """
        node_set = set(nodes)

        # Guard: every edge endpoint must be in the declared node list
        for edge in edges:
            if edge.source not in node_set:
                raise ValueError(
                    f"DoCalculusSimulator: edge source '{edge.source}' is not in the nodes list."
                )
            if edge.target not in node_set:
                raise ValueError(
                    f"DoCalculusSimulator: edge target '{edge.target}' is not in the nodes list."
                )

        # Guard: every intervention node must be in the declared node list
        for node in interventions:
            if node not in node_set:
                raise ValueError(
                    f"DoCalculusSimulator: Intervention node '{node}' is not in the nodes list."
                )

        # Seed predictions with intervention values (std_dev=0: we know these exactly)
        predictions: Dict[str, GaussianPrediction] = {
            node: GaussianPrediction(mean=value, std_dev=0.0)
            for node, value in interventions.items()
        }

        # Track original incoming edges to identify source nodes
        orig_incoming: Dict[str, List[str]] = {node: [] for node in nodes}
        for edge in edges:
            orig_incoming[edge.target].append(edge.source)

        # Do-calculus: cut all incoming edges to intervened nodes
        active_edges = [e for e in edges if e.target not in interventions]

        # Build adjacency list of incoming (source, weight) per node, and outgoing targets
        incoming_edges: Dict[str, List[Tuple[str, float]]] = {node: [] for node in nodes}
        outgoing_edges: Dict[str, List[str]] = {node: [] for node in nodes}
        for edge in active_edges:
            incoming_edges[edge.target].append((edge.source, edge.weight))
            outgoing_edges[edge.source].append(edge.target)

        # Iterative topological sort (Kahn's algorithm) — no recursion, detects cycles
        in_degree = {node: len(incoming_edges[node]) for node in nodes}
        queue = deque([node for node in nodes if in_degree[node] == 0])
        topo_order: List[str] = []

        while queue:
            current_node = queue.popleft()
            topo_order.append(current_node)
            for target in outgoing_edges[current_node]:
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        # If not all nodes were processed, the graph has a cycle
        if len(topo_order) != len(nodes):
            unprocessed = [n for n in nodes if n not in topo_order]
            raise ValueError(
                f"DoCalculusSimulator: cycle detected in graph. "
                f"Nodes involved: {unprocessed}. Causal graphs must be acyclic (DAGs)."
            )

        # Forward propagation
        for node in topo_order:
            if node in predictions:
                continue

            parent_preds = []
            edge_weights = []

            for parent, weight in incoming_edges[node]:
                if parent in predictions:
                    parent_preds.append({
                        "mean": predictions[parent].mean,
                        "std_dev": predictions[parent].std_dev,
                    })
                    edge_weights.append(weight)
                else:
                    # Parent was not predicted — log a warning, do not silently skip
                    warnings.warn(
                        f"DoCalculusSimulator: parent '{parent}' of node '{node}' "
                        "has no prediction yet. It will be excluded from this node's calculation. "
                        "Check graph connectivity.",
                        UserWarning,
                        stacklevel=2,
                    )

            if parent_preds:
                child_pred = self.gp_engine.predict_child(parent_preds, edge_weights)
                predictions[node] = GaussianPrediction(
                    mean=child_pred["mean"],
                    std_dev=child_pred["std_dev"],
                )
            else:
                is_source = len(orig_incoming[node]) == 0
                if is_source:
                    default_val = getattr(node, "default_value", 0.0)
                    predictions[node] = GaussianPrediction(
                        mean=default_val,
                        std_dev=self.gp_engine.base_noise,
                    )
                    warnings.warn(
                        f"DoCalculusSimulator: non-intervened source '{node}' defaulted to {default_val}.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        f"DoCalculusSimulator: node '{node}' has no predicted parents. "
                        "Skipping prediction — downstream nodes will correctly report high uncertainty.",
                        UserWarning,
                        stacklevel=2,
                    )

        return SimulationStepOutput(predictions=predictions)
