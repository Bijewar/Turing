import logging
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class GESGraphBuilder:
    """
    Step 3 (DATA PATH): Converts a pristine DataFrame into a mathematical
    causal graph using the Greedy Equivalence Search (GES) Algorithm.
    GES scales much better to high-dimensional datasets (>50 variables) than PC.
    """

    @staticmethod
    def build_graph(df: pd.DataFrame, alpha: float = 0.05) -> dict:
        """
        Runs the GES algorithm on the DataFrame.
        Returns a networkx compatible JSON dictionary.
        """
        if df.empty:
            raise ValueError("Cannot build graph from empty DataFrame.")

        columns = df.columns.tolist()
        if len(columns) < 2:
            raise ValueError("GES Algorithm requires at least two columns.")

        data_matrix = df.to_numpy(dtype=float)

        logger.info(f"Running GES Algorithm on {len(df)} rows, {len(columns)} columns")

        try:
            from causallearn.search.ScoreBased.GES import ges
        except ImportError:
            raise ImportError("causal-learn is not installed. GES cannot run.")

        # Run GES (Greedy Equivalence Search) algorithm
        # We use the default score function (local score, e.g., BIC)
        cg = ges(data_matrix)

        # Convert causal-learn graph to NetworkX
        nx_graph = nx.DiGraph()

        # Add all nodes first
        for col in columns:
            nx_graph.add_node(col, id=col, label=col, type="variable", confidence=100.0)

        # In GES, cg['G'] is the GeneralGraph object containing the adjacency matrix
        if isinstance(cg, dict) and 'G' in cg:
            adj_matrix = cg['G'].graph
        elif hasattr(cg, 'G'):
            adj_matrix = cg.G.graph
        else:
            adj_matrix = np.zeros((len(columns), len(columns)), dtype=int)

        num_nodes = len(columns)
        corr_matrix = np.corrcoef(data_matrix, rowvar=False) if num_nodes > 1 else np.array([[1.0]])

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue

                # Check for directed edge i -> j (causal-learn uses -1 for source and 1 for target)
                if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    correlation = float(corr_matrix[i, j]) if num_nodes > 1 else 0.0
                    if np.isnan(correlation):
                        correlation = 0.0
                        
                    confidence = min(100.0, max(0.0, float(abs(correlation) * 100.0)))
                    if confidence == 0.0:
                        confidence = 75.0

                    nx_graph.add_edge(
                        columns[i],
                        columns[j],
                        weight=correlation,
                        confidence=confidence,
                        relation="CAUSES"
                    )

        return GESGraphBuilder._serialize_nx(nx_graph)

    @staticmethod
    def _serialize_nx(graph: nx.DiGraph) -> dict:
        nodes = []
        for node, data in graph.nodes(data=True):
            nodes.append({"id": node, **data})

        edges = []
        for source, target, data in graph.edges(data=True):
            edges.append({"source": source, "target": target, **data})

        return {
            "nodes": nodes,
            "edges": edges
        }
