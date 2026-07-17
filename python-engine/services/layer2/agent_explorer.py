import os
import random
import warnings
from typing import Dict, Optional
from schemas.layer2 import SearchSpace, AgentProposal


class AgentExplorer:
    """
    Explorer Agent.
    Takes the Bayesian base point and pushes each variable to the farthest
    boundary of its search space, deliberately seeking the edges of the system
    to find where performance degrades (cliff detection).
    """

    def __init__(self, seed: Optional[int] = None):
        _seed_env = os.getenv("AGENT_RANDOM_STATE") or os.getenv("GP_RANDOM_STATE")
        if seed is not None:
            self._seed = seed
        elif _seed_env is not None:
            self._seed = int(_seed_env)
        else:
            self._seed = None
        self._rng = random.Random(self._seed)

    def propose(
        self,
        base_point: Dict[str, float],
        domain_config: Dict[str, SearchSpace],
    ) -> AgentProposal:
        """
        Propose extreme boundary values for every controllable variable.

        Args:
            base_point:    {node_name: current_best_value} from BayesianOptimizer
            domain_config: {node_name: SearchSpace(min, max)}

        Returns:
            AgentProposal with values pushed to the farthest boundary per variable.
        """
        # Re-check env var for dynamically mocked test environments
        _seed_env = os.getenv("AGENT_RANDOM_STATE") or os.getenv("GP_RANDOM_STATE")
        if _seed_env is not None:
            rng = random.Random(int(_seed_env))
        else:
            rng = self._rng

        if not domain_config and not base_point:
            warnings.warn(
                "AgentExplorer: both base_point and domain_config are empty. "
                "Returning empty proposal.",
                UserWarning,
                stacklevel=2,
            )
            return AgentProposal(
                agent_name="Explorer",
                proposed_values={},
                justification="No variables to explore (empty config and base_point).",
            )

        proposed = {}
        push_details = []

        all_nodes = set(base_point.keys()).union(domain_config.keys())

        for node in all_nodes:
            space = domain_config.get(node)

            if space is None:
                val = base_point[node]
                warnings.warn(
                    f"AgentExplorer: node '{node}' has no SearchSpace in domain_config. "
                    "Falling back to raw base_point value — no boundary push applied.",
                    UserWarning,
                    stacklevel=2,
                )
                proposed[node] = val
                push_details.append(f"{node}={val} (no space — not pushed)")
                continue

            # Default to a random plausible value if the node is missing from base_point
            val = base_point.get(node, rng.uniform(space.min, space.max))

            if val < space.min or val > space.max:
                warnings.warn(
                    f"AgentExplorer: value for '{node}' ({val}) is outside "
                    f"the domain config [{space.min}, {space.max}].",
                    UserWarning,
                    stacklevel=2,
                )

            dist_to_min = abs(val - space.min)
            dist_to_max = abs(val - space.max)

            if dist_to_min > dist_to_max:
                # Further from min — push to min (the farther extreme)
                proposed[node] = space.min
                push_details.append(f"{node}={val}->{space.min} (pushed to MIN)")
            elif dist_to_max > dist_to_min:
                # Further from max — push to max (the farther extreme)
                proposed[node] = space.max
                push_details.append(f"{node}={val}->{space.max} (pushed to MAX)")
            else:
                # Exactly at center — documented tie-break: always push to min
                proposed[node] = space.min
                push_details.append(f"{node}={val}->{space.min} (center tie -> pushed to MIN)")

        justification = (
            "Explorer pushed variables to their farthest boundaries to find cliff edges. "
            "Decisions: " + "; ".join(push_details) + "."
        )

        return AgentProposal(
            agent_name="Explorer",
            proposed_values=proposed,
            justification=justification,
        )
