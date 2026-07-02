"""
services/layer1/gaussian_process.py — Gaussian Process Tools

Combines the Gaussian Process confidence estimator (main) for Layer 1
with the GPEngine mathematical predictor (Sub_Manas) for Layer 2 simulation.
"""
import os
import math
import warnings
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from schemas.graph import CausalGraph, Node


# ==============================================================================
# 1. LAYER 2 GP PREDICTION ENGINE (Sub_Manas)
# ==============================================================================

class GPEngine:
    """
    Gaussian Process Engine.
    Uses scikit-learn's GaussianProcessRegressor with RBF and White kernels
    to mathematically predict causal propagation and compound uncertainty.
    """
    RANDOM_STATE = int(os.getenv("GP_RANDOM_STATE", "42"))

    def __init__(self, base_noise: float = 0.02):
        self.base_noise = base_noise
        self.random_state = int(os.getenv("GP_RANDOM_STATE", str(self.RANDOM_STATE)))

    def predict_child(self, parent_predictions: List[Dict[str, float]], edge_weights: List[Optional[float]]) -> Dict[str, float]:
        """
        Propagates uncertainty through a real Gaussian Process fitted on the fly.
        """
        import warnings
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        if not parent_predictions:
            warnings.warn(
                "GPEngine.predict_child: called with no parents. Returning base noise fallback.",
                UserWarning,
                stacklevel=2
            )
            return {"mean": 0.0, "std_dev": self.base_noise}

        if len(parent_predictions) != len(edge_weights):
            raise ValueError(
                f"GPEngine.predict_child: parent_predictions length ({len(parent_predictions)}) "
                f"does not match edge_weights length ({len(edge_weights)})."
            )

        sanitized_weights = []
        for i, w in enumerate(edge_weights):
            if w is None:
                warnings.warn(
                    f"GPEngine: edge_weights[{i}] is None — defaulting to 0.5. ",
                    UserWarning,
                    stacklevel=2
                )
                sanitized_weights.append(0.5)
            else:
                sanitized_weights.append(w)

        # Generate local synthetic points representing the parent distributions
        n_samples = 50
        X_train = np.zeros((n_samples, len(parent_predictions)))
        y_train = np.zeros(n_samples)

        seed = int(os.getenv("GP_RANDOM_STATE", str(self.RANDOM_STATE)))
        rng = np.random.default_rng(seed)

        for i in range(len(parent_predictions)):
            mean = parent_predictions[i]["mean"]
            std = parent_predictions[i]["std_dev"]
            weight = sanitized_weights[i]
            
            # Sample from the parent distribution
            X_train[:, i] = rng.normal(mean, max(std, 1e-4), n_samples)
            # Propagate expected linear causal mechanism
            y_train += X_train[:, i] * weight

        # Apply structural noise
        y_train += rng.normal(0, self.base_noise, n_samples)

        # Fit a real Gaussian Process to map the parent space to the child space
        kernel = 1.0 * RBF(length_scale=np.ones(len(parent_predictions))) + WhiteKernel(noise_level=self.base_noise**2)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)
        
        try:
            gp.fit(X_train, y_train)
            X_test = np.array([[p["mean"] for p in parent_predictions]])
            child_mean, child_std = gp.predict(X_test, return_std=True)
            return {"mean": round(float(child_mean[0]), 4), "std_dev": round(float(child_std[0]), 4)}
        except Exception as e:
            # Fallback to mean quadrature only if the GP matrix inversion fails
            warnings.warn(f"GP Fit failed, falling back: {e}")
            child_mean = sum(p["mean"] * w for p, w in zip(parent_predictions, sanitized_weights))
            child_std = math.sqrt(sum(p["std_dev"] ** 2 for p in parent_predictions) + self.base_noise ** 2)
            return {"mean": round(float(child_mean), 4), "std_dev": round(child_std, 4)}


# ==============================================================================
# 2. LAYER 1 CONFIDENCE REFINEMENT (main)
# ==============================================================================

def refine_confidence_with_gp(graph: CausalGraph) -> CausalGraph:
    """
    Uses graph topology (in/out degree) and a Gaussian Process Regressor
    to refine edge/node confidence scores, smoothing out noisy estimates.
    """
    try:
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        features, raw_confidences = _build_feature_matrix(graph)

        if len(features) < 2:
            return graph  # Not enough data points for GP fitting

        X = np.array(features, dtype=float)
        y = np.array(raw_confidences, dtype=float)

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-2)
        gp.fit(X, y)

        y_pred, _ = gp.predict(X, return_std=True)

        # Clamp GP predictions to [5, 100]
        refined_nodes = []
        for node, pred in zip(graph.nodes, y_pred):
            refined_conf = float(max(5.0, min(100.0, pred)))
            node_dict = node.model_dump()
            node_dict["confidence"] = round(refined_conf, 1)
            refined_nodes.append(Node(**node_dict))

        return CausalGraph(nodes=refined_nodes, edges=graph.edges)

    except ImportError as e:
        # ERR-B36 fix: scikit-learn not installed — raise exception instead of silently failing
        raise ImportError("scikit-learn is required for Gaussian Process confidence refinement") from e


def _build_feature_matrix(graph: CausalGraph):
    """
    Builds a feature matrix from graph topology:
    [in_degree, out_degree, degree_ratio, raw_confidence]
    """
    in_degree = {node.id: 0 for node in graph.nodes}
    out_degree = {node.id: 0 for node in graph.nodes}

    for edge in graph.edges:
        if edge.source in out_degree:
            out_degree[edge.source] += 1
        if edge.target in in_degree:
            in_degree[edge.target] += 1

    features = []
    raw_confidences = []

    for node in graph.nodes:
        ind = in_degree.get(node.id, 0)
        outd = out_degree.get(node.id, 0)
        total = ind + outd
        ratio = outd / total if total > 0 else 0.5

        features.append([float(ind), float(outd), ratio])
        raw_confidences.append(node.confidence)

    return features, raw_confidences


# ==============================================================================
# 3. STRUCTURAL FITTER
# ==============================================================================

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "models")

class StructuralFitter:
    """
    Step 5: Fits Gaussian Process equations to causal graphs.
    Only processes DATA path graphs. Skips TEXT path.
    """

    @staticmethod
    def fit_graph(df: pd.DataFrame, graph_data: dict, path_type: str) -> dict:
        """
        Fits a GP for every target node based on its causal parents.
        Saves the models to disk and returns augmented graph data.
        """
        import logging
        import uuid
        import joblib
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        logger = logging.getLogger(__name__)

        if path_type == "TEXT":
            logger.info("Skipping Structural Equation Fitter for TEXT path.")
            graph_data["is_fitted"] = False
            return graph_data

        if df.empty:
            raise ValueError("Cannot fit graph with empty DataFrame.")

        logger.info(f"Fitting Structural Equations for DATA path graph with {len(df)} rows...")

        session_id = str(uuid.uuid4())
        models = {}

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        parents_map = {n["id"]: [] for n in nodes}
        for e in edges:
            target = e["target"]
            source = e["source"]
            if target in parents_map:
                parents_map[target].append(source)

        global_warnings = []

        for node in nodes:
            node_id = node["id"]
            parents = parents_map.get(node_id, [])

            if not parents:
                node["fit_metrics"] = {"r2": None, "mean_uncertainty": None, "status": "SOURCE"}
                continue

            valid_parents = [p for p in parents if p in df.columns]
            if not valid_parents or node_id not in df.columns:
                node["fit_metrics"] = {"r2": None, "mean_uncertainty": None, "status": "MISSING_DATA"}
                continue

            X = df[valid_parents].values
            y = df[node_id].values

            x_stds = np.std(X, axis=0) if X.size else np.ones(len(valid_parents))
            y_std = float(np.std(y)) if y.size else 1.0
            init_length_scale = [max(float(s), 1e-3) for s in x_stds]
            init_noise_level  = max(y_std * 0.1, 1e-4)

            kernel = (
                1.0 * RBF(length_scale=init_length_scale)
                + WhiteKernel(noise_level=init_noise_level)
            )

            _rs_env = os.getenv("GP_RANDOM_STATE")
            random_state = int(_rs_env) if _rs_env is not None else None

            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=random_state,
            )

            try:
                gp.fit(X, y)
                models[node_id] = {"model": gp, "parents": valid_parents}

                r2 = gp.score(X, y)
                y_pred, std = gp.predict(X, return_std=True)
                mean_uncertainty = float(np.mean(std))

                example_parent_vals = [f"{p}={X[0][i]:.2f}" for i, p in enumerate(valid_parents)]
                example_str = f"{', '.join(example_parent_vals)} → {node_id}={y_pred[0]:.2f} ± {std[0]:.2f}"

                sparse_warning = False
                y_std = float(np.std(y))
                if y_std > 0 and (mean_uncertainty / y_std) > 0.2:
                    sparse_warning = True
                    warning_msg = f"Sparse data detected for {node_id}. High uncertainty relative to variance in causal relationship."
                    global_warnings.append(warning_msg)
                    node.setdefault("warnings", []).append(warning_msg)

                node["fit_metrics"] = {
                    "r2": float(r2),
                    "mean_uncertainty": mean_uncertainty,
                    "status": "FITTED",
                    "example_equation": example_str,
                    "sparse_data_warning": sparse_warning
                }
                logger.debug(f"Fitted GP for {node_id} (R2: {r2:.3f}, Unc: {mean_uncertainty:.3f})")

            except Exception as e:
                logger.error(f"Failed to fit GP for node {node_id}: {e}")
                node["fit_metrics"] = {"r2": None, "mean_uncertainty": None, "status": f"ERROR: {str(e)}"}

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{session_id}.joblib")
        try:
            joblib.dump(models, model_path)
            logger.info(f"Successfully saved {len(models)} models to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save models to disk: {e}")
            raise RuntimeError(f"Model serialization failed: {e}")

        graph_data["is_fitted"] = True
        graph_data["session_id"] = session_id
        if global_warnings:
            graph_data.setdefault("global_warnings", []).extend(global_warnings)

        return graph_data