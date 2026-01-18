"""Orchestrator configuration management."""

import yaml
from typing import Dict, Any, List
from pathlib import Path


class OrchestratorConfig:
    """Manage orchestrator configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize orchestrator configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.orchestrator_config = self.config.get('orchestrator', {})
        
        # Service dependency definitions
        self.service_dependencies = {
            'llm_client': [],  # LLM client can be initialized early
            'data_loader': [],
            'data_preprocessor': ['data_loader'],
            'feature_engineer': ['data_preprocessor'],
            'misconception_detector': [],
            'knowledge_graph_constructor': ['data_preprocessor', 'misconception_detector'],
            'knowledge_graph_query': ['knowledge_graph_constructor'],
            'competency_predictor': [],
            'difficulty_calibrator': [],
            'question_recommender': ['competency_predictor', 'difficulty_calibrator', 'knowledge_graph_query'],
            'learning_path_generator': ['competency_predictor', 'knowledge_graph_query'],
            'gap_analyzer': ['competency_predictor', 'knowledge_graph_query'],
            'misconception_analyzer': ['misconception_detector', 'knowledge_graph_query'],
            'training_pipeline': ['data_preprocessor', 'feature_engineer'],
            'model_evaluator': [],
            'model_exporter': []
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.orchestrator_config.get(key, default)
    
    def get_service_dependencies(self, service_name: str) -> List[str]:
        """
        Get dependencies for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            List of dependency service names
        """
        return self.service_dependencies.get(service_name, [])
    
    def get_initialization_order(self) -> List[str]:
        """
        Get service initialization order based on dependencies.
        
        Returns:
            List of service names in initialization order
        """
        # Topological sort based on dependencies
        visited = set()
        result = []
        
        def visit(service: str):
            if service in visited:
                return
            visited.add(service)
            for dep in self.service_dependencies.get(service, []):
                visit(dep)
            result.append(service)
        
        for service in self.service_dependencies.keys():
            visit(service)
        
        return result
