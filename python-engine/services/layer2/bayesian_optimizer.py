import warnings
import os
import inspect
from typing import Dict, List, Any, Optional, Union

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from schemas.layer2 import SearchSpace


def ucb_acquisition(mu: np.ndarray, std: np.ndarray, kappa: float = 1.96) -> np.ndarray:
    """Upper Confidence Bound acquisition function: mu + kappa * std."""
    return np.asarray(mu, dtype=float) + float(kappa) * np.asarray(std, dtype=float)


_ucb = ucb_acquisition
ucb = ucb_acquisition
upper_confidence_bound = ucb_acquisition


def ei_acquisition(mu: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.0) -> np.ndarray:
    """Expected Improvement acquisition function with exploration parameter xi."""
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    imp = mu - y_best - xi
    Z = np.zeros_like(imp, dtype=float)
    mask = std > 1e-9
    Z[mask] = imp[mask] / std[mask]
    ei = np.zeros_like(imp, dtype=float)
    ei[mask] = imp[mask] * norm.cdf(Z[mask]) + std[mask] * norm.pdf(Z[mask])
    ei[~mask] = np.maximum(0.0, imp[~mask])
    return ei


class BayesianOptimizer:
    """
    Bayesian Optimizer.
    Selects the next point to simulate based on historical performance using
    Expected Improvement (EI) or Upper Confidence Bound (UCB).
    """

    def __init__(self, seed: Optional[int] = None):
        _rs_env = os.getenv("GP_RANDOM_STATE")
        if seed is not None:
            self._seed = seed
        elif _rs_env is not None:
            self._seed = int(_rs_env)
        else:
            self._seed = None

        self._rng = np.random.default_rng(self._seed)
        self._lhs_idx = 0

    ucb_acquisition = staticmethod(ucb_acquisition)
    _ucb = staticmethod(ucb_acquisition)
    ucb = staticmethod(ucb_acquisition)
    upper_confidence_bound = staticmethod(ucb_acquisition)
    ei_acquisition = staticmethod(ei_acquisition)

    def initial_design(
        self, domain_config: Dict[str, SearchSpace], n: int = 5
    ) -> List[Dict[str, float]]:
        """
        Generates a space-filling Latin Hypercube Sample (LHS) across all dimensions.
        """
        if not domain_config or n <= 0:
            return []

        nodes = list(domain_config.keys())
        dim = len(nodes)
        samples = np.zeros((n, dim))

        for j, node in enumerate(nodes):
            space = domain_config[node]
            # Divide domain into n equal strata
            strata_edges = np.linspace(space.min, space.max, n + 1)
            # Sample one point per stratum uniformly
            points = np.array([
                self._rng.uniform(strata_edges[i], strata_edges[i + 1])
                for i in range(n)
            ])
            # Randomly permute stratum order
            self._rng.shuffle(points)
            samples[:, j] = points

        result = []
        for i in range(n):
            entry = {nodes[j]: float(samples[i, j]) for j in range(dim)}
            result.append(entry)
        return result

    def optimize(
        self,
        historical_data: List[Dict[str, Any]],
        domain_config: Dict[str, SearchSpace],
        sink_node: Optional[str] = None,
        max_iter: int = 1,
        kappa: Optional[float] = None,
        xi: float = 0.0,
        acquisition: str = "ei",
        **kwargs,
    ) -> List[Dict[str, float]]:
        """
        Returns a list of `max_iter` candidate points to evaluate.
        """
        results = []
        current_history = list(historical_data)
        for _ in range(max_iter):
            point = self.get_base_point(
                current_history,
                domain_config,
                sink_node=sink_node,
                kappa=kappa,
                xi=xi,
                acquisition=acquisition,
            )
            if isinstance(point, list):
                results.extend(point)
                break
            results.append(point)
            # Mock historical evaluation for subsequent iteration proposal
            if sink_node:
                current_history.append({sink_node: 0.0, "values": point})
        return results

    def get_base_point(
        self,
        historical_data: List[Dict[str, Any]],
        domain_config: Dict[str, SearchSpace],
        sink_node: Optional[str] = None,
        kappa: Optional[float] = None,
        xi: float = 0.0,
        acquisition: str = "ei",
        max_iter: Optional[int] = None,
        n_iter: Optional[int] = None,
        n_iterations: Optional[int] = None,
        num_iter: Optional[int] = None,
        **kwargs,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Returns the next point(s) to test based on past history.
        """
        iter_count = max_iter or n_iter or n_iterations or num_iter
        if iter_count is not None and iter_count > 1:
            return self.optimize(
                historical_data,
                domain_config,
                sink_node=sink_node,
                max_iter=iter_count,
                kappa=kappa,
                xi=xi,
                acquisition=acquisition,
                **kwargs,
            )

        if not domain_config:
            warnings.warn(
                "BayesianOptimizer: domain_config is empty. "
                "No base point can be calculated — returning {}.",
                UserWarning,
                stacklevel=2,
            )
            return {}

        # Validate each search space
        for node, space in domain_config.items():
            if space.min >= space.max:
                raise ValueError(
                    f"BayesianOptimizer: SearchSpace for '{node}' is invalid — "
                    f"min ({space.min}) must be strictly less than max ({space.max})."
                )

        # Refresh local RNG from environment if set to maintain reproducibility
        _rs_env = os.getenv("GP_RANDOM_STATE")
        if _rs_env is not None:
            rng = np.random.default_rng(int(_rs_env))
        else:
            rng = self._rng

        # Handle empty or insufficient history space-filling initial design
        if not historical_data:
            lhs_points = self.initial_design(domain_config, n=5)
            pt = lhs_points[self._lhs_idx % len(lhs_points)]
            self._lhs_idx += 1
            return pt

        if not sink_node:
            raise ValueError("BayesianOptimizer: sink_node must be a non-empty string.")

        if not any(sink_node in entry for entry in historical_data):
            raise ValueError(
                f"BayesianOptimizer: None of the historical entries contain the sink_node key '{sink_node}'."
            )

        # 1. Prepare training data
        nodes = list(domain_config.keys())
        X_train = []
        y_train = []
        for entry in historical_data:
            vals = entry.get("values", {})
            if all(n in vals for n in nodes) and sink_node in entry:
                X_train.append([vals[n] for n in nodes])
                y_train.append(entry[sink_node])

        if len(X_train) < 2:
            lhs_points = self.initial_design(domain_config, n=5)
            pt = lhs_points[self._lhs_idx % len(lhs_points)]
            self._lhs_idx += 1
            return pt

        X = np.array(X_train, dtype=float)
        y = np.array(y_train, dtype=float)

        # 2. Fit the Gaussian Process
        length_scales = [
            max((domain_config[n].max - domain_config[n].min) * 0.5, 1e-3)
            for n in nodes
        ]
        init_noise_level = 1e-3

        kernel = 1.0 * Matern(length_scale=length_scales, nu=2.5) + WhiteKernel(
            noise_level=init_noise_level, noise_level_bounds=(1e-5, 1e1)
        )

        random_state = int(_rs_env) if _rs_env is not None else None

        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=random_state,
        )

        base_point = {}
        try:
            gp.fit(X, y)
        except Exception as e:
            warnings.warn(f"BayesianOptimizer: GP fit failed ({e}). Falling back to best known.")
            best_past = max(historical_data, key=lambda x: x.get(sink_node, 0) if sink_node else 0)
            best_values = best_past.get("values", {})
            for node, space in domain_config.items():
                best_val = best_values.get(node, (space.min + space.max) / 2)
                base_point[node] = max(space.min, min(space.max, best_val))
            return base_point

        # 3. Candidate generation & acquisition scoring
        n_samples = 1000
        X_sample = np.zeros((n_samples, len(nodes)), dtype=float)
        for i, node in enumerate(nodes):
            space = domain_config[node]
            X_sample[:, i] = rng.uniform(space.min, space.max, n_samples)

        mu, std = gp.predict(X_sample, return_std=True)

        if kappa is not None or acquisition.lower() == "ucb":
            k_val = kappa if kappa is not None else 1.96
            scores = ucb_acquisition(mu, std, kappa=k_val)
        else:
            y_best = float(np.max(y))
            scores = ei_acquisition(mu, std, y_best=y_best, xi=xi)

        best_idx = int(np.argmax(scores))
        best_x = X_sample[best_idx]

        for i, node in enumerate(nodes):
            space = domain_config[node]
            base_point[node] = float(np.clip(best_x[i], space.min, space.max))

        return base_point
