import networkx as nx
import json
import os
from typing import List, Dict, Any, Tuple
from networkx.readwrite import json_graph

class KnowledgeGraph:
    def __init__(self, storage_file: str = "kg.json"):
        self.storage_file = storage_file
        self.graph = nx.DiGraph()
        self._load_graph()

    def _load_graph(self):
        """Load graph from JSON file if it exists."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                self.graph = json_graph.node_link_graph(data)
            except Exception as e:
                print(f"Error loading graph: {e}")

    def _save_graph(self):
        """Save graph to JSON file."""
        try:
            data = json_graph.node_link_data(self.graph)
            with open(self.storage_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving graph: {e}")

    def add_relation(self, subject: str, predicate: str, object_: str):
        """Add a triplet to the graph."""
        self.graph.add_edge(subject, object_, relation=predicate)
        self._save_graph()

    def get_subgraph(self, nodes: List[str], depth: int = 1) -> List[Dict[str, Any]]:
        """Get a subgraph centered around specific nodes."""
        subgraph_nodes = set(nodes)
        for node in nodes:
            if self.graph.has_node(node):
                # Get neighbors within depth
                # Simple implementation for depth=1
                neighbors = self.graph.neighbors(node)
                subgraph_nodes.update(neighbors)
        
        # Extract edges
        edges = []
        for u, v in self.graph.subgraph(subgraph_nodes).edges():
            data = self.graph.get_edge_data(u, v)
            edges.append({
                "subject": u,
                "predicate": data.get("relation", "related_to"),
                "object": v
            })
        return edges

    def search_nodes(self, query: str) -> List[str]:
        """Find nodes that match the query (simple substring match for now)."""
        # In a real system, this would use vector search on node names
        return [node for node in self.graph.nodes() if query.lower() in node.lower()]

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """Return all relations in the graph."""
        edges = []
        for u, v in self.graph.edges():
            data = self.graph.get_edge_data(u, v)
            edges.append({
                "subject": u,
                "predicate": data.get("relation", "related_to"),
                "object": v
            })
        return edges
