"""
FILE: python-engine/services/layer1/classifier.py
PURPOSE: Step 6 - Node Classifier. Analyzes graph topology and mathematically assigns semantic roles (Source, Sink, Bottleneck) and UI themes.
"""
import logging
import json
from typing import Dict, Any
from litellm import completion

logger = logging.getLogger(__name__)

class NodeClassifier:
    """
    Step 6: Classifies graph nodes purely based on topology.
    """

    @staticmethod
    def classify_graph(graph_data: dict, model_name: str = "gpt-4o") -> dict:
        """
        Calculates in-degree and out-degree for every node and assigns roles.
        """
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            return graph_data

        logger.info(f"Classifying {len(nodes)} nodes based on topology...")

        # Calculate degrees
        in_degrees = {node["id"]: 0 for node in nodes}
        out_degrees = {node["id"]: 0 for node in nodes}

        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            
            if source in out_degrees:
                out_degrees[source] += 1
            if target in in_degrees:
                in_degrees[target] += 1

        # Apply classification
        for node in nodes:
            node_id = node["id"]
            indeg = in_degrees.get(node_id, 0)
            outdeg = out_degrees.get(node_id, 0)

            # Default attributes
            node["is_hidden"] = node.get("is_hidden", False)
            node["border_style"] = "dotted" if node["is_hidden"] else "solid"
            node["is_pulsing"] = False
            
            if indeg == 0:
                # No incoming edges
                node["node_class"] = "SOURCE"
                node["theme"] = "blue"
                node["semantic_label"] = "Researcher controls"
            elif outdeg == 0:
                # No outgoing edges
                node["node_class"] = "SINK"
                node["theme"] = "green"
                node["semantic_label"] = "Outcome variable"
            elif indeg > 1 and outdeg >= 1:
                # Multiple inputs, at least one output
                node["node_class"] = "BOTTLENECK"
                node["theme"] = "red"
                node["semantic_label"] = "Single point of failure"
                node["is_pulsing"] = True
            else:
                # indeg == 1 and outdeg >= 1
                node["node_class"] = "MEDIATOR"
                node["theme"] = "white"
                node["semantic_label"] = "Cannot directly control"

            logger.debug(f"Node {node_id} classified as {node['node_class']} (in:{indeg}, out:{outdeg})")

        # Hidden Node Detection Pass
        if edges:
            logger.info("Running Agentic Hidden Node Detection Pass...")
            prompt = f"""
            You are a Causal Graph analyzer. Look at these causal edges:
            {json.dumps([{"source": e["source"], "target": e["target"]} for e in edges], indent=2)}
            
            Look for "gaps" in the causal chains where a step from A to C logically implies a missing intermediate node B.
            Identify up to 3 major missing intermediate nodes.
            Return ONLY a valid JSON array of objects.
            Schema: [{{"source": "A", "target": "C", "hidden_node_name": "B"}}]
            If no gaps are found, return an empty array [].
            """
            
            try:
                import os
                response = completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} if "gpt" in model_name or "claude" in model_name else None,
                )
                content = response.choices[0].message.content.strip()
                import re
                match = re.search(r'\[.*?\]', content, re.DOTALL)
                if match:
                    content = match.group(0)
                    
                gaps = json.loads(content)
                if isinstance(gaps, dict):
                    for v in gaps.values():
                        if isinstance(v, list):
                            gaps = v
                            break
                            
                if isinstance(gaps, list):
                    for gap in gaps:
                        if not isinstance(gap, dict): continue
                        u = gap.get("source")
                        v = gap.get("target")
                        hidden_name = gap.get("hidden_node_name")
                        
                        if u and v and hidden_name:
                            hidden_node_id = str(hidden_name).upper()
                            # Create hidden node if it doesn't exist
                            if not any(n["id"] == hidden_node_id for n in nodes):
                                nodes.append({
                                    "id": hidden_node_id,
                                    "label": hidden_name,
                                    "type": "entity",
                                    "is_hidden": True,
                                    "border_style": "dotted",
                                    "node_class": "HIDDEN",
                                    "theme": "gray",
                                    "semantic_label": "Inferred missing node",
                                    "confidence": 50.0
                                })
                            # Create inferred edges
                            edges.append({
                                "source": u,
                                "target": hidden_node_id,
                                "relation": "INFERRED",
                                "confidence": 50.0,
                                "weight": 0.5
                            })
                            edges.append({
                                "source": hidden_node_id,
                                "target": v,
                                "relation": "INFERRED",
                                "confidence": 50.0,
                                "weight": 0.5
                            })
                            logger.info(f"AGENT ADDED HIDDEN NODE: {hidden_node_id} between {u} and {v}")
            except Exception as e:
                logger.error(f"Hidden node detection failed: {e}")

        return graph_data
