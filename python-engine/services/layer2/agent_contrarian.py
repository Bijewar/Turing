import os
import random
import warnings
from typing import Dict, Optional
from schemas.layer2 import SearchSpace, AgentProposal


class AgentContrarian:
    """
    Contrarian Agent.
    Challenges assumptions by picking the boundary that is OPPOSITE to what
    the Explorer would choose — ensuring every round covers a unique boundary
    and no simulation round is wasted on a duplicate proposal.
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
        Propose boundary values that directly challenge the current base point.

        Args:
            base_point:    {node_name: current_best_value} from BayesianOptimizer
            domain_config: {node_name: SearchSpace(min, max)}

        Returns:
            AgentProposal with boundary values opposite to what Explorer would pick.
        """
        _seed_env = os.getenv("AGENT_RANDOM_STATE") or os.getenv("GP_RANDOM_STATE")
        if _seed_env is not None:
            rng = random.Random(int(_seed_env))
        else:
            rng = self._rng

        if not domain_config and not base_point:
            warnings.warn(
                "AgentContrarian: both base_point and domain_config are empty. "
                "Returning empty proposal.",
                UserWarning,
                stacklevel=2,
            )
            return AgentProposal(
                agent_name="Contrarian",
                proposed_values={},
                justification="No variables to challenge (empty config and base_point).",
            )

        proposed = {}
        challenge_details = []

        all_nodes = set(base_point.keys()).union(domain_config.keys())

        for node in all_nodes:
            space = domain_config.get(node)

            if space is None:
                val = base_point[node]
                warnings.warn(
                    f"AgentContrarian: node '{node}' has no SearchSpace in domain_config. "
                    "Falling back to raw base_point value — no boundary challenge applied.",
                    UserWarning,
                    stacklevel=2,
                )
                proposed[node] = val
                challenge_details.append(f"{node}={val} (no space — not challenged)")
                continue

            val = base_point.get(node, rng.uniform(space.min, space.max))

            dist_to_min = abs(val - space.min)
            dist_to_max = abs(val - space.max)

            # Anti-duplication: Explorer always picks the FARTHER boundary.
            # Contrarian always picks the OPPOSITE (closer) boundary to guarantee
            # that no two agents test the same corner in the same round.
            # Tie-break: Explorer goes to min, so Contrarian goes to max.
            if dist_to_min > dist_to_max:
                chosen = space.max
                challenge_details.append(
                    f"{node}={val}->MAX({space.max}) [opposite of Explorer's MIN]"
                )
            elif dist_to_max > dist_to_min:
                chosen = space.min
                challenge_details.append(
                    f"{node}={val}->MIN({space.min}) [opposite of Explorer's MAX]"
                )
            else:
                chosen = space.max
                challenge_details.append(
                    f"{node}={val}->MAX({space.max}) [center tie -> opposite of Explorer's MIN]"
                )

            proposed[node] = chosen

        justification = (
            "Contrarian challenged Explorer's assumptions by testing the opposite boundaries. "
            "Challenges: " + "; ".join(challenge_details) + "."
        )

        return AgentProposal(
            agent_name="Contrarian",
            proposed_values=proposed,
            justification=justification,
        )
