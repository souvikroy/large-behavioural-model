"""Knowledge graph embeddings."""

import networkx as nx
import numpy as np
from typing import Dict, Optional, List
from node2vec import Node2Vec


class GraphEmbedder:
    """Generate embeddings for knowledge graph nodes."""
    
    def __init__(
        self,
        graph: nx.Graph,
        dimensions: int = 128,
        walk_length: int = 30,
        num_walks: int = 200,
        window_size: int = 10
    ):
        """
        Initialize graph embedder.
        
        Args:
            graph: NetworkX graph
            dimensions: Embedding dimensions
            walk_length: Length of random walks
            num_walks: Number of walks per node
            window_size: Context window size
        """
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.model = None
        self.embeddings = {}
    
    def generate_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Generate node embeddings using Node2Vec.
        
        Returns:
            Dictionary mapping node IDs to embeddings
        """
        # Convert to undirected for Node2Vec (if needed)
        if isinstance(self.graph, nx.DiGraph):
            graph_undirected = self.graph.to_undirected()
        else:
            graph_undirected = self.graph
        
        # Initialize Node2Vec
        node2vec = Node2Vec(
            graph_undirected,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=1
        )
        
        # Train model
        self.model = node2vec.fit(window=self.window_size, min_count=1, batch_words=4)
        
        # Get embeddings
        self.embeddings = {
            node: self.model.wv[node] 
            for node in self.graph.nodes()
        }
        
        return self.embeddings
    
    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node embedding or None
        """
        if node_id in self.embeddings:
            return self.embeddings[node_id]
        return None
    
    def get_similar_nodes(self, node_id: str, top_k: int = 10) -> List[tuple]:
        """
        Get most similar nodes to a given node.
        
        Args:
            node_id: Node identifier
            top_k: Number of similar nodes to return
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        if self.model is None or node_id not in self.model.wv:
            return []
        
        similar = self.model.wv.most_similar(node_id, topn=top_k)
        return similar
