"""Knowledge graph visualization utilities."""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List, Set
import pyvis
from pyvis.network import Network


class KnowledgeGraphVisualizer:
    """Visualize knowledge graph."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize visualizer.
        
        Args:
            graph: Knowledge graph to visualize
        """
        self.graph = graph
    
    def visualize_subgraph(
        self,
        nodes: List[str],
        output_path: Optional[str] = None,
        layout: str = 'spring'
    ) -> None:
        """
        Visualize a subgraph.
        
        Args:
            nodes: List of node identifiers to include
            output_path: Path to save visualization (optional)
            layout: Layout algorithm ('spring', 'circular', 'hierarchical')
        """
        # Create subgraph
        subgraph = self.graph.subgraph(nodes)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        elif layout == 'hierarchical':
            pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot')
        else:
            pos = nx.spring_layout(subgraph)
        
        # Color nodes by type
        node_colors = []
        for node in subgraph.nodes():
            node_type = self.graph.nodes[node].get('node_type', 'unknown')
            color_map = {
                'grade': 'red',
                'subject': 'orange',
                'chapter': 'yellow',
                'learning_objective': 'green',
                'learning_unit': 'blue',
                'question': 'purple',
                'concept': 'cyan',
                'misconception': 'pink',
                'bloom_level': 'gray'
            }
            node_colors.append(color_map.get(node_type, 'black'))
        
        # Draw graph
        nx.draw(
            subgraph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_size=8,
            arrows=True,
            edge_color='gray'
        )
        
        plt.title('Knowledge Graph Visualization')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_interactive(
        self,
        output_path: str = 'knowledge_graph.html',
        height: str = '800px',
        width: str = '100%'
    ) -> None:
        """
        Create interactive visualization using Pyvis.
        
        Args:
            output_path: Path to save HTML file
            height: Height of visualization
            width: Width of visualization
        """
        # Create Pyvis network
        net = Network(height=height, width=width, directed=True)
        
        # Add nodes
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            label = data.get('name', node)
            
            # Set node properties based on type
            color_map = {
                'grade': '#FF0000',
                'subject': '#FF8800',
                'chapter': '#FFDD00',
                'learning_objective': '#00FF00',
                'learning_unit': '#0088FF',
                'question': '#8800FF',
                'concept': '#00FFFF',
                'misconception': '#FF00FF',
                'bloom_level': '#888888'
            }
            
            net.add_node(
                node,
                label=label[:30],  # Truncate long labels
                color=color_map.get(node_type, '#000000'),
                title=f"{node_type}: {label}"
            )
        
        # Add edges
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            weight = data.get('weight', 1.0)
            
            net.add_edge(
                u,
                v,
                title=edge_type,
                value=weight
            )
        
        # Configure physics
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        # Save
        net.save_graph(output_path)
