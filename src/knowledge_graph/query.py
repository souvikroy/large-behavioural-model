"""Knowledge graph query and traversal utilities."""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import deque


class KnowledgeGraphQuery:
    """Query and traverse knowledge graph."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize graph query engine.
        
        Args:
            graph: Knowledge graph
        """
        self.graph = graph
    
    def find_prerequisites(self, concept_node: str, max_depth: int = 5) -> List[str]:
        """
        Find all prerequisites for a concept (transitive closure).
        
        Args:
            concept_node: Concept node identifier
            max_depth: Maximum depth to traverse
            
        Returns:
            List of prerequisite node identifiers
        """
        if concept_node not in self.graph:
            return []
        
        prerequisites = set()
        queue = deque([(concept_node, 0)])
        visited = set()
        
        while queue:
            current, depth = queue.popleft()
            
            if depth >= max_depth or current in visited:
                continue
            
            visited.add(current)
            
            # Find incoming PREREQUISITE edges
            for u, v, data in self.graph.in_edges(current, data=True):
                if data.get('edge_type') == 'PREREQUISITE':
                    prerequisites.add(u)
                    if u not in visited:
                        queue.append((u, depth + 1))
        
        return list(prerequisites)
    
    def find_learning_path(
        self,
        from_concept: str,
        to_concept: str,
        respect_prerequisites: bool = True
    ) -> List[str]:
        """
        Find learning path between two concepts.
        
        Args:
            from_concept: Starting concept node
            to_concept: Target concept node
            respect_prerequisites: Whether to respect prerequisite relationships
            
        Returns:
            List of nodes in the path
        """
        if from_concept not in self.graph or to_concept not in self.graph:
            return []
        
        # Use BFS to find shortest path
        try:
            if respect_prerequisites:
                # Only follow PREREQUISITE and PROGRESSES_TO edges
                subgraph = nx.DiGraph()
                for u, v, data in self.graph.edges(data=True):
                    if data.get('edge_type') in ['PREREQUISITE', 'PROGRESSES_TO', 'CONTAINS']:
                        subgraph.add_edge(u, v)
                
                if from_concept in subgraph and to_concept in subgraph:
                    path = nx.shortest_path(subgraph, from_concept, to_concept)
                    return path
            else:
                # Use all edges
                path = nx.shortest_path(self.graph, from_concept, to_concept)
                return path
        except nx.NetworkXNoPath:
            return []
    
    def find_misconception_root_causes(self, misconception_node: str) -> List[str]:
        """
        Trace misconceptions to root concepts.
        
        Args:
            misconception_node: Misconception node identifier
            
        Returns:
            List of root concept nodes
        """
        if misconception_node not in self.graph:
            return []
        
        root_causes = []
        
        # Find concepts associated with misconception
        for u, v, data in self.graph.edges(misconception_node, data=True):
            if data.get('edge_type') == 'ASSOCIATED_WITH':
                # Find prerequisites of this concept
                prerequisites = self.find_prerequisites(v)
                root_causes.extend(prerequisites)
        
        return list(set(root_causes))
    
    def find_similar_concepts(
        self,
        concept_node: str,
        max_distance: int = 3
    ) -> List[Tuple[str, int]]:
        """
        Find similar concepts using graph distance.
        
        Args:
            concept_node: Concept node identifier
            max_distance: Maximum graph distance
            
        Returns:
            List of (concept_node, distance) tuples
        """
        if concept_node not in self.graph:
            return []
        
        # Use BFS to find nodes within distance
        distances = {}
        queue = deque([(concept_node, 0)])
        visited = set([concept_node])
        
        while queue:
            current, distance = queue.popleft()
            
            if distance > max_distance:
                continue
            
            distances[current] = distance
            
            # Traverse edges
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        # Filter to concept nodes only
        similar = [
            (node, dist) 
            for node, dist in distances.items()
            if node != concept_node and self.graph.nodes[node].get('node_type') == 'concept'
        ]
        
        return sorted(similar, key=lambda x: x[1])
    
    def filter_by_competency(
        self,
        concept_nodes: List[str],
        min_competency: float = 0.7
    ) -> List[str]:
        """
        Filter concepts by required competency level.
        
        Args:
            concept_nodes: List of concept node identifiers
            min_competency: Minimum competency threshold
            
        Returns:
            Filtered list of concept nodes
        """
        filtered = []
        
        for node in concept_nodes:
            if node not in self.graph:
                continue
            
            # Check if node has competency requirement
            node_data = self.graph.nodes[node]
            if 'competency_score' in node_data:
                if node_data['competency_score'] is not None:
                    if node_data['competency_score'] >= min_competency:
                        filtered.append(node)
            else:
                # If no competency data, include it
                filtered.append(node)
        
        return filtered
    
    def get_concept_details(self, concept_node: str) -> Dict:
        """
        Get detailed information about a concept.
        
        Args:
            concept_node: Concept node identifier
            
        Returns:
            Dictionary with concept details
        """
        if concept_node not in self.graph:
            return {}
        
        node_data = self.graph.nodes[concept_node].copy()
        
        # Add related information
        node_data['prerequisites'] = self.find_prerequisites(concept_node)
        node_data['related_concepts'] = self.find_similar_concepts(concept_node, max_distance=2)
        
        # Find associated misconceptions
        misconceptions = []
        for u, v, data in self.graph.edges(concept_node, data=True):
            if data.get('edge_type') == 'ASSOCIATED_WITH' and self.graph.nodes[u].get('node_type') == 'misconception':
                misconceptions.append(u)
        
        node_data['misconceptions'] = misconceptions
        
        return node_data
