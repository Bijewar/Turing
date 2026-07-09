import warnings
from typing import List, Tuple, Union, Optional
import numpy as np

from schemas.layer2 import HeatmapPoint, ZoneFinderInput, ZoneFinderOutput, UnexploredZone, SearchSpace
from services.layer2.do_calculus import DoCalculusSimulator


class CliffDetector:
    """
    Detects cliffs (sharp drop-offs) in the parameter space.
    """

    @staticmethod
    def detect(grid: Union[np.ndarray, List[Any]], threshold: float = 3.0) -> List[Tuple[int, int]]:
        """
        Detects steep gradients / cliffs in a 2D surface grid.

        Args:
            grid: 2D numpy array or list of lists representing the outcome landscape.
            threshold: Minimum absolute difference between adjacent cells to be flagged as a cliff.

        Returns:
            List of (row, col) tuples where cliffs were detected.
        """
        arr = np.asarray(grid, dtype=float)
        if arr.size == 0:
            raise ValueError("CliffDetector.detect: input grid is empty.")

        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return []

        diff_r = np.abs(np.diff(arr, axis=0))
        diff_c = np.abs(np.diff(arr, axis=1))

        cliff_coords = set()

        r_indices, c_indices = np.where(diff_r >= threshold)
        for r, c in zip(r_indices, c_indices):
            cliff_coords.add((int(r), int(c)))
            cliff_coords.add((int(r + 1), int(c)))

        r_indices_c, c_indices_c = np.where(diff_c >= threshold)
        for r, c in zip(r_indices_c, c_indices_c):
            cliff_coords.add((int(r), int(c)))
            cliff_coords.add((int(r), int(c + 1)))

        return sorted(list(cliff_coords))

    @staticmethod
    def detect_cliffs(data_points: List[HeatmapPoint], cliff_sigma: float = 1.5) -> None:
        """
        Flags points as cliffs if their z_val is significantly lower than the mean.
        Modifies data_points in place.
        """
        if not data_points:
            return

        z_vals = [pt.z_val for pt in data_points]
        mean_z = sum(z_vals) / len(z_vals)

        if len(z_vals) > 1:
            variance = sum((z - mean_z) ** 2 for z in z_vals) / (len(z_vals) - 1)
        else:
            variance = 0.0

        std_z = variance ** 0.5

        if std_z < 1e-9:
            for pt in data_points:
                pt.is_cliff = False
        else:
            cliff_threshold = mean_z - cliff_sigma * std_z
            for pt in data_points:
                pt.is_cliff = pt.z_val < cliff_threshold


class UnexploredZoneFinder:
    """
    Checks original data against simulated space to find mathematically
    uncertain but highly promising unexplored zones.
    """

    def __init__(self):
        self.simulator = DoCalculusSimulator()

    def find_zones(self, payload: ZoneFinderInput) -> ZoneFinderOutput:
        unexplored_zones = []

        # ── Step 1: Determine Source Nodes and Sink Node ────────────────────
        in_degrees = {node: 0 for node in payload.nodes}
        out_degrees = {node: 0 for node in payload.nodes}

        for edge in payload.edges:
            if edge.source in out_degrees:
                out_degrees[edge.source] += 1
            if edge.target in in_degrees:
                in_degrees[edge.target] += 1

        source_nodes = [
            node for node in payload.nodes
            if in_degrees[node] == 0 and out_degrees[node] > 0
        ]

        if not source_nodes:
            raise ValueError(
                "UnexploredZoneFinder: no source nodes found in the graph "
                "(nodes with no incoming edges and at least one outgoing edge)."
            )

        # Resolve sink node dynamically
        if payload.sink_node:
            sink_node = payload.sink_node
            if sink_node not in payload.nodes:
                raise ValueError(
                    f"UnexploredZoneFinder: specified sink_node '{sink_node}' "
                    "is not in the nodes list."
                )
        else:
            sink_candidates = [
                n for n in payload.nodes
                if out_degrees[n] == 0 and in_degrees[n] > 0
            ]
            if sink_candidates:
                sink_node = sink_candidates[-1]
            else:
                sink_node = payload.nodes[-1]
                warnings.warn(
                    f"UnexploredZoneFinder: could not auto-detect sink node. "
                    f"Falling back to last node: '{sink_node}'.",
                    UserWarning,
                    stacklevel=2,
                )

        # ── Step 2: Extract historically tested ranges for each source node ──
        tested_ranges = {
            node: {"min": float('inf'), "max": -float('inf')}
            for node in source_nodes
        }

        # Find best past run for hold-steady logic later
        best_past = {}
        if payload.historical_data:
            if not any(sink_node in entry for entry in payload.historical_data):
                warnings.warn(
                    f"UnexploredZoneFinder: None of the historical entries contain the sink_node key '{sink_node}'. "
                    "Hold-steady logic will fall back to domain center values.",
                    UserWarning,
                    stacklevel=2,
                )
            best_run = max(
                payload.historical_data,
                key=lambda x: x.get(sink_node, 0)
            )
            best_past = best_run.get("values", {})

        for data_point in payload.historical_data:
            values = data_point.get("values", {})
            for node in source_nodes:
                val = values.get(node)
                if val is not None:
                    tested_ranges[node]["min"] = min(tested_ranges[node]["min"], val)
                    tested_ranges[node]["max"] = max(tested_ranges[node]["max"], val)

        # ── Step 3: Compare tested ranges against domain config boundaries ────
        for node in source_nodes:
            space = payload.domain_config.get(node)
            if not space:
                warnings.warn(
                    f"UnexploredZoneFinder: node '{node}' has no SearchSpace in "
                    "domain_config. Skipping zone detection for this parameter.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            tested_min = tested_ranges[node]["min"]
            tested_max = tested_ranges[node]["max"]

            # If no data for this node at all, entire space is unexplored
            if tested_min == float('inf'):
                unexplored_gaps = [(space.min, space.max)]
            else:
                unexplored_gaps = []
                space_width = space.max - space.min
                gap_size_required = space_width * payload.gap_threshold

                # Check bottom gap
                if tested_min - space.min > gap_size_required:
                    unexplored_gaps.append((space.min, tested_min))
                # Check top gap
                if space.max - tested_max > gap_size_required:
                    unexplored_gaps.append((tested_max, space.max))

            # ── Step 4: Simulate midpoint of unexplored gaps ──────────────────
            for gap_min, gap_max in unexplored_gaps:
                midpoint = (gap_min + gap_max) / 2

                interventions = {node: midpoint}
                for other_node in source_nodes:
                    if other_node != node:
                        other_space = payload.domain_config.get(other_node)
                        if other_space:
                            if other_node in best_past:
                                steady_val = best_past[other_node]
                            else:
                                steady_val = (other_space.min + other_space.max) / 2
                        else:
                            steady_val = 0.0
                        interventions[other_node] = steady_val

                sim_res = self.simulator.simulate(
                    payload.nodes, payload.edges, interventions
                )

                pred = sim_res.predictions.get(sink_node)
                if not pred:
                    continue

                num_experiments = len(payload.historical_data)

                msg = (
                    f"You have never tested {node} between {round(gap_min, 1)} and {round(gap_max, 1)}. "
                    f"Simulation predicts {sink_node} may reach {round(pred.mean, 1)} "
                    f"but uncertainty is high ± {round(pred.std_dev, 1)} — "
                    f"only {num_experiments} experiment(s) globally."
                )

                unexplored_zones.append(UnexploredZone(
                    parameter_name=node,
                    range_min=round(gap_min, 2),
                    range_max=round(gap_max, 2),
                    predicted_yield=round(pred.mean, 2),
                    uncertainty=round(pred.std_dev, 2),
                    message=msg
                ))

        return ZoneFinderOutput(unexplored_zones=unexplored_zones)
