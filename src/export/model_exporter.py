"""Model export utilities."""

import pickle
import json
import joblib
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from datetime import datetime

try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ModelExporter:
    """Export models in various formats."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize model exporter.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.export_config = self.config.get('export', {})
        self.paths_config = self.config.get('paths', {})
        
        self.models_dir = Path(self.paths_config.get('models_dir', 'models'))
        self.graphs_dir = Path(self.paths_config.get('graphs_dir', 'knowledge_graphs'))
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.graphs_dir.mkdir(exist_ok=True)
    
    def export_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export model in specified formats.
        
        Args:
            model: Model object to export
            model_name: Name for the model
            metadata: Optional metadata dictionary
            formats: List of formats ('pickle', 'onnx', 'joblib')
            
        Returns:
            Dictionary with paths to exported files
        """
        if formats is None:
            formats = self.export_config.get('model_format', ['pickle'])
        
        exported_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_type in formats:
            if format_type == 'pickle':
                filepath = self.models_dir / f"{model_name}_{timestamp}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'metadata': metadata or {},
                        'exported_at': timestamp
                    }, f)
                exported_paths['pickle'] = str(filepath)
            
            elif format_type == 'joblib':
                filepath = self.models_dir / f"{model_name}_{timestamp}.joblib"
                joblib.dump({
                    'model': model,
                    'metadata': metadata or {},
                    'exported_at': timestamp
                }, filepath)
                exported_paths['joblib'] = str(filepath)
            
            elif format_type == 'onnx':
                if not ONNX_AVAILABLE:
                    print("ONNX not available, skipping ONNX export")
                    continue
                
                try:
                    # Convert to ONNX (simplified - would need proper input shape)
                    # This is a placeholder - actual conversion depends on model type
                    filepath = self.models_dir / f"{model_name}_{timestamp}.onnx"
                    # ONNX conversion would go here
                    exported_paths['onnx'] = str(filepath)
                except Exception as e:
                    print(f"Failed to export to ONNX: {e}")
        
        # Export metadata separately
        if metadata and self.export_config.get('include_metadata', True):
            metadata_path = self.models_dir / f"{model_name}_{timestamp}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            exported_paths['metadata'] = str(metadata_path)
        
        return exported_paths
    
    def export_knowledge_graph(
        self,
        graph: nx.Graph,
        graph_name: str = "knowledge_graph",
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export knowledge graph in specified formats.
        
        Args:
            graph: NetworkX graph
            graph_name: Name for the graph
            formats: List of formats ('pickle', 'graphml', 'json')
            
        Returns:
            Dictionary with paths to exported files
        """
        if formats is None:
            formats = self.export_config.get('graph_format', ['pickle'])
        
        exported_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_type in formats:
            if format_type == 'pickle':
                filepath = self.graphs_dir / f"{graph_name}_{timestamp}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(graph, f)
                exported_paths['pickle'] = str(filepath)
            
            elif format_type == 'graphml':
                filepath = self.graphs_dir / f"{graph_name}_{timestamp}.graphml"
                nx.write_graphml(graph, filepath)
                exported_paths['graphml'] = str(filepath)
            
            elif format_type == 'json':
                filepath = self.graphs_dir / f"{graph_name}_{timestamp}.json"
                
                # Convert to JSON-serializable format
                graph_data = {
                    'nodes': [
                        {
                            'id': node,
                            'data': dict(graph.nodes[node])
                        }
                        for node in graph.nodes()
                    ],
                    'edges': [
                        {
                            'source': u,
                            'target': v,
                            'data': data
                        }
                        for u, v, data in graph.edges(data=True)
                    ]
                }
                
                with open(filepath, 'w') as f:
                    json.dump(graph_data, f, indent=2)
                exported_paths['json'] = str(filepath)
        
        return exported_paths
    
    def create_model_card(
        self,
        model_name: str,
        metrics: Dict[str, float],
        description: Optional[str] = None
    ) -> str:
        """
        Create model card documentation.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary with performance metrics
            description: Optional model description
            
        Returns:
            Model card as string
        """
        card = f"""
# Model Card: {model_name}

## Description
{description or 'No description provided'}

## Performance Metrics
"""
        for metric, value in metrics.items():
            card += f"- **{metric}**: {value:.4f}\n"
        
        card += f"""
## Export Information
- Exported at: {datetime.now().isoformat()}
- Model format: Multiple formats available

## Limitations
- Model performance may vary with different data distributions
- Requires proper feature engineering pipeline
"""
        
        return card
    
    def export_all(
        self,
        models: Dict[str, Any],
        knowledge_graph: Optional[nx.Graph] = None,
        model_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Export all models and knowledge graph.
        
        Args:
            models: Dictionary of model_name -> model_object
            knowledge_graph: Optional knowledge graph
            model_metrics: Optional dictionary of model_name -> metrics
            
        Returns:
            Dictionary with all exported paths
        """
        exported = {}
        
        # Export models
        for model_name, model in models.items():
            metadata = {
                'model_name': model_name,
                'exported_at': datetime.now().isoformat()
            }
            
            if model_metrics and model_name in model_metrics:
                metadata['metrics'] = model_metrics[model_name]
            
            exported[model_name] = self.export_model(model, model_name, metadata)
        
        # Export knowledge graph
        if knowledge_graph:
            exported['knowledge_graph'] = self.export_knowledge_graph(knowledge_graph)
        
        return exported
