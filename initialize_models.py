"""Initialize models for API - uses orchestrator for service management."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.orchestrator import Orchestrator
import yaml

def initialize_models():
    """Initialize models using orchestrator."""
    print("Initializing models using orchestrator...")
    
    try:
        # Load config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Get data path
        data_path = config.get('data', {}).get('input_file', 'question-response.json')
        
        # Initialize orchestrator
        orchestrator = Orchestrator()
        init_result = orchestrator.initialize(data_path)
        
        print(f"Orchestrator initialized: {init_result.get('services_initialized', 0)} services")
        print(f"Health status: {init_result.get('health', {}).get('overall', 'unknown')}")
        
        # Get services from orchestrator
        services = orchestrator.get_all_services()
        
        return {
            'competency_predictor': services.get('competency_predictor'),
            'difficulty_calibrator': services.get('difficulty_calibrator'),
            'question_recommender': services.get('question_recommender'),
            'learning_path_generator': services.get('learning_path_generator'),
            'gap_analyzer': services.get('gap_analyzer'),
            'misconception_analyzer': services.get('misconception_analyzer'),
            'knowledge_graph_query': services.get('knowledge_graph_query'),
            'df_processed': orchestrator.service_manager.processed_data,
            'orchestrator': orchestrator  # Also return orchestrator for advanced usage
        }
        
    except Exception as e:
        print(f"Error initializing models with orchestrator: {e}")
        print("Falling back to minimal initialization...")
        
        # Fallback: create minimal orchestrator instance
        try:
            orchestrator = Orchestrator()
            orchestrator.initialize()  # Initialize without data
            
            services = orchestrator.get_all_services()
            
            return {
                'competency_predictor': services.get('competency_predictor'),
                'difficulty_calibrator': services.get('difficulty_calibrator'),
                'question_recommender': services.get('question_recommender'),
                'learning_path_generator': services.get('learning_path_generator'),
                'gap_analyzer': services.get('gap_analyzer'),
                'misconception_analyzer': services.get('misconception_analyzer'),
                'knowledge_graph_query': services.get('knowledge_graph_query'),
                'df_processed': None,
                'orchestrator': orchestrator
            }
        except Exception as fallback_error:
            print(f"Fallback initialization also failed: {fallback_error}")
            return {
                'competency_predictor': None,
                'difficulty_calibrator': None,
                'question_recommender': None,
                'learning_path_generator': None,
                'gap_analyzer': None,
                'misconception_analyzer': None,
                'knowledge_graph_query': None,
                'df_processed': None,
                'orchestrator': None
            }

if __name__ == "__main__":
    models = initialize_models()
    print("\nModels ready for API use!")
