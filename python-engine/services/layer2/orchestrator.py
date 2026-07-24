import asyncio
import warnings
from typing import List, Dict, Any, Optional

from schemas.graph import CausalGraph, Node, Edge
from schemas.layer2 import (
    Layer2Request, Layer2Response, IterationResult, AgentAction,
    RoundInput, GraphEdge, SearchSpace
)
from services.layer2.agent_explorer import AgentExplorer
from services.layer2.agent_exploiter import AgentExploiter
from services.layer2.agent_contrarian import AgentContrarian
from services.layer2.do_calculus import DoCalculusSimulator
from services.layer2.bayesian_optimizer import BayesianOptimizer


async def run_bayesian_optimization(request: Layer2Request) -> Layer2Response:
    """
    Orchestration loop that runs max_iterations of the Bayesian Optimization cycle
    using three ReAct agents (Explorer, Exploiter, Contrarian).
    """
    orch = Orchestrator(
        graph=request.graph,
        target_node_id=request.target_node_id,
        max_rounds=request.max_iterations,
        domain_config=request.domain_config,
    )
    return await orch.run()


class Orchestrator:
    """
    Orchestrator class interface for Layer 2 agent simulation and Bayesian Optimization.
    """

    def __init__(
        self,
        graph: Optional[CausalGraph] = None,
        target_node_id: Optional[str] = None,
        max_rounds: int = 10,
        max_iterations: Optional[int] = None,
        domain_config: Optional[Dict[str, SearchSpace]] = None,
        **kwargs,
    ):
        self.graph = graph or CausalGraph(
            nodes=[Node(id="A", label="A", confidence=80.0), Node(id="B", label="B", confidence=80.0)],
            edges=[Edge(source="A", target="B", relation="ACTIVATES", confidence=80.0)],
        )
        self.max_rounds = max_iterations if max_iterations is not None else max_rounds
        self.max_iterations = self.max_rounds
        self.user_domain_config = domain_config or {}

        # Resolve target node as a valid sink
        node_ids = [n.id for n in self.graph.nodes]
        out_degrees = {n: 0 for n in node_ids}
        in_degrees = {n: 0 for n in node_ids}
        for e in self.graph.edges:
            if e.source in out_degrees:
                out_degrees[e.source] += 1
            if e.target in in_degrees:
                in_degrees[e.target] += 1

        sink_candidates = [n for n in node_ids if out_degrees[n] == 0 and in_degrees[n] > 0]
        if not sink_candidates:
            sink_candidates = [n for n in node_ids if out_degrees[n] == 0]

        if target_node_id and target_node_id in node_ids:
            self.target_node_id = target_node_id
        elif sink_candidates:
            self.target_node_id = sink_candidates[-1]
        else:
            self.target_node_id = node_ids[-1]

    run_bayesian_optimization = staticmethod(run_bayesian_optimization)
    simulate = staticmethod(run_bayesian_optimization)

    async def run(self) -> Layer2Response:
        """
        Executes the Bayesian optimization iterations.
        """
        optimizer = BayesianOptimizer()
        explorer = AgentExplorer()
        exploiter = AgentExploiter()
        contrarian = AgentContrarian()
        simulator = DoCalculusSimulator()

        nodes = [node.id for node in self.graph.nodes]
        edges_list = [
            GraphEdge(source=e.source, target=e.target, weight=e.weight, confidence=e.confidence)
            for e in self.graph.edges
        ]

        in_degrees = {n: 0 for n in nodes}
        for e in self.graph.edges:
            in_degrees[e.target] += 1

        source_nodes = [n for n in nodes if in_degrees[n] == 0]

        # Build dynamic domain config without hardcoded values
        domain_config = {}
        for n in source_nodes:
            if n in self.user_domain_config:
                domain_config[n] = self.user_domain_config[n]
            else:
                domain_config[n] = SearchSpace(min=-10.0, max=10.0)

        historical_data: List[Dict[str, Any]] = []
        simulation_results: List[IterationResult] = []

        best_intervention_str = "None"
        overall_confidence = 0.0

        for i in range(self.max_rounds):
            await asyncio.sleep(0)  # Cooperate with event loop / allow cancellation

            base_point = optimizer.get_base_point(
                historical_data, domain_config, sink_node=self.target_node_id
            )

            proposals = [
                explorer.propose(base_point, domain_config),
                exploiter.propose(historical_data, domain_config, sink_node=self.target_node_id),
                contrarian.propose(base_point, domain_config),
            ]

            async def simulate_proposal(prop):
                res = await asyncio.to_thread(
                    simulator.simulate, nodes, edges_list, prop.proposed_values
                )
                return prop, res

            sim_tasks = [simulate_proposal(p) for p in proposals]
            results = await asyncio.gather(*sim_tasks, return_exceptions=True)

            best_agent = None
            best_score = -float('inf')
            best_yield = 0.0
            best_values = {}

            agent_actions = []
            new_round_data = []

            for item in results:
                if isinstance(item, Exception):
                    continue
                prop, sim_res = item
                pred = sim_res.predictions.get(self.target_node_id)
                if not pred:
                    continue

                # Expected Improvement / UCB ranking metric
                score = float(pred.mean - (0.5 * pred.std_dev))

                agent_actions.append(AgentAction(
                    agent_id=prop.agent_name,
                    agent_role=prop.agent_name.lower().replace("agent ", ""),
                    think=prop.justification,
                    decide=str(prop.proposed_values),
                    act=f"Yield Prediction: {pred.mean:.4f} ± {pred.std_dev:.4f}",
                    expected_improvement=round(score, 4),
                ))

                # Record all proposals into training data to avoid wasting simulation trials
                new_round_data.append({
                    self.target_node_id: float(pred.mean),
                    "values": dict(prop.proposed_values),
                })

                if score > best_score:
                    best_score = score
                    best_agent = prop.agent_name
                    best_yield = float(pred.mean)
                    best_values = dict(prop.proposed_values)

            historical_data.extend(new_round_data)

            if best_agent is not None:
                best_intervention_str = str(best_values)
                overall_confidence = max(0.0, min(100.0, float(best_score * 10)))

                simulation_results.append(IterationResult(
                    iteration=i + 1,
                    agent_actions=agent_actions,
                    best_intervention=best_intervention_str,
                    predicted_outcome=best_yield,
                    confidence=overall_confidence,
                ))
            else:
                break

        return Layer2Response(
            final_graph=self.graph,
            simulation_results=simulation_results,
            best_intervention=best_intervention_str,
            confidence=round(overall_confidence, 2),
        )
