"""Knowledge graph construction utilities."""

import networkx as nx
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from .builder import KnowledgeGraphBuilder


class KnowledgeGraphConstructor:
    """Construct and manage knowledge graph."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize knowledge graph constructor.
        
        Args:
            config_path: Path to configuration file
        """
        self.builder = KnowledgeGraphBuilder(config_path)
        self.graph = None
    
    def construct(
        self, 
        df: pd.DataFrame, 
        misconception_detector=None
    ) -> nx.MultiDiGraph:
        """
        Construct knowledge graph from data.
        
        Args:
            df: DataFrame with educational data
            misconception_detector: Optional MisconceptionDetector instance
            
        Returns:
            Constructed knowledge graph
        """
        self.graph = self.builder.build(df, misconception_detector)
        return self.graph
    
    def get_node_info(self, node_id: str) -> Dict:
        """
        Get information about a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dictionary with node information
        """
        if self.graph is None or node_id not in self.graph:
            return {}
        
        return self.graph.nodes[node_id]
    
    def get_edges_by_type(self, edge_type: str) -> List[Tuple]:
        """
        Get all edges of a specific type.
        
        Args:
            edge_type: Type of edge
            
        Returns:
            List of (source, target) tuples
        """
        if self.graph is None:
            return []
        
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == edge_type:
                edges.append((u, v))
        
        return edges
    
    def get_prerequisites(self, concept_node: str) -> List[str]:
        """
        Get prerequisite concepts for a concept.
        
        Args:
            concept_node: Concept node identifier
            
        Returns:
            List of prerequisite node identifiers
        """
        if self.graph is None or concept_node not in self.graph:
            return []
        
        prerequisites = []
        for u, v, data in self.graph.in_edges(concept_node, data=True):
            if data.get('edge_type') == 'PREREQUISITE':
                prerequisites.append(u)
        
        return prerequisites
    
    def get_related_concepts(self, concept_node: str) -> List[str]:
        """
        Get related concepts.
        
        Args:
            concept_node: Concept node identifier
            
        Returns:
            List of related concept node identifiers
        """
        if self.graph is None or concept_node not in self.graph:
            return []
        
        related = []
        for u, v, data in self.graph.edges(concept_node, data=True):
            if data.get('edge_type') in ['RELATED_TO', 'PART_OF']:
                related.append(v)
        
        return related
