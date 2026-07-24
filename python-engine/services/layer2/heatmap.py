import warnings
from typing import List, Dict, Any, Optional
import numpy as np

from schemas.layer2 import (
    HeatmapInput, HeatmapOutput, HeatmapPoint, SearchSpace, GraphEdge, GaussianPrediction
)
from services.layer2.do_calculus import DoCalculusSimulator
from services.layer2.cliff_detector import CliffDetector


def linspace(start: float, stop: float, num: int) -> List[float]:
    """Return `num` evenly spaced values from start to stop (inclusive)."""
    if num <= 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [float(start + step * i) for i in range(num)]


class HeatmapGenerator:
    """
    Generates a full 2D heatmap of the parameter space using the GP simulator.
    If there are >2 source nodes, picks the first two for axes and holds the
    rest steady at their historically best values (hold-steady logic).
    """

    def __init__(self):
        self.simulator = DoCalculusSimulator()

    @classmethod
    def generate(cls, payload: HeatmapInput) -> HeatmapOutput:
        """
        Classmethod / instance-method support for generating heatmap grid.
        """
        instance = cls()
        return instance._generate(payload)

    def _generate(self, payload: HeatmapInput) -> HeatmapOutput:
        nodes = list(payload.nodes)
        edges = list(payload.edges)
        domain_config = dict(payload.domain_config)

        # Handle minimal/empty default input
        if not nodes:
            nodes = ["x", "y"]
            edges = [GraphEdge(source="x", target="y")]
            if "x" not in domain_config:
                domain_config["x"] = SearchSpace(min=0.0, max=10.0)
            if "y" not in domain_config:
                domain_config["y"] = SearchSpace(min=0.0, max=10.0)

        # ── Step 1: Determine source nodes from graph topology ──────────────
        in_degrees = {node: 0 for node in nodes}
        out_degrees = {node: 0 for node in nodes}

        for edge in edges:
            if edge.source in out_degrees:
                out_degrees[edge.source] += 1
            if edge.target in in_degrees:
                in_degrees[edge.target] += 1

        source_nodes = [
            node for node in nodes
            if in_degrees[node] == 0 and out_degrees[node] > 0
        ]

        if not source_nodes:
            source_nodes = [nodes[0]]

        # ── Step 2: Select X and Y axes ──────────────────────────────────────
        x_label = source_nodes[0]
        y_label = source_nodes[1] if len(source_nodes) > 1 else (nodes[1] if len(nodes) > 1 else source_nodes[0])
        single_axis = (x_label == y_label)

        # Ensure domain_config has entries for axis nodes
        for label in set([x_label, y_label]):
            if label not in domain_config:
                domain_config[label] = SearchSpace(min=0.0, max=10.0)

        x_space = domain_config[x_label]
        y_space = domain_config[y_label]

        # ── Step 3: Resolve sink node ─────────────────────────────────────────
        if payload.sink_node:
            sink_node_name = payload.sink_node
            if sink_node_name not in nodes:
                sink_node_name = nodes[-1]
        else:
            sink_candidates = [
                n for n in nodes
                if out_degrees.get(n, 0) == 0 and in_degrees.get(n, 0) > 0
            ]
            if sink_candidates:
                sink_node_name = sink_candidates[-1]
            else:
                sink_node_name = nodes[-1]

        # ── Step 4: Hold-steady values for extra source nodes ─────────────────
        steady_values = {}
        if len(source_nodes) > 2:
            best_past = {}
            if payload.historical_data:
                matching_runs = [
                    x for x in payload.historical_data if sink_node_name in x
                ]
                if matching_runs:
                    best_run = max(matching_runs, key=lambda x: x.get(sink_node_name, 0))
                    best_past = best_run.get("values", {})

            for node in source_nodes[2:]:
                if node not in domain_config:
                    domain_config[node] = SearchSpace(min=0.0, max=100.0)
                space = domain_config[node]
                steady_values[node] = best_past.get(node, (space.min + space.max) / 2)

        # ── Step 5: Generate grid ─────────────────────────────────────────────
        resolution = max(2, min(200, payload.resolution))
        x_steps = linspace(x_space.min, x_space.max, resolution)
        y_steps = linspace(y_space.min, y_space.max, resolution)
        data_points = []

        for x_val in x_steps:
            for y_val in y_steps:
                interventions = {x_label: float(x_val)}
                if x_label != y_label:
                    interventions[y_label] = float(y_val)
                interventions.update(steady_values)

                try:
                    sim_res = self.simulator.simulate(nodes, edges, interventions)
                    pred = sim_res.predictions.get(sink_node_name)
                    mean_val = float(pred.mean) if pred else float(x_val * 0.5 + y_val * 0.5)
                except Exception:
                    mean_val = float(x_val * 0.5 + y_val * 0.5)

                data_points.append(HeatmapPoint(
                    x_val=round(float(x_val), 2),
                    y_val=round(float(y_val), 2),
                    z_val=round(mean_val, 2),
                    is_cliff=False,
                    value=round(mean_val, 2),
                    color=max(0.0, min(1.0, mean_val / 100.0 if mean_val > 1.0 else (mean_val if mean_val >= 0 else 0.0))),
                ))

        # ── Step 6: Relative cliff detection ─────────────────────────────────
        CliffDetector.detect_cliffs(data_points, payload.cliff_sigma)

        return HeatmapOutput(
            x_label=x_label,
            y_label=y_label,
            is_1d=single_axis,
            data=data_points,
        )
