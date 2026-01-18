"""Service manager for lifecycle and dependency management."""

import logging
from typing import Dict, Any, Optional, List
import yaml
import pandas as pd

from .config import OrchestratorConfig

# Import all services
from ..data.loader import load_dataframe
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..models.competency_predictor import CompetencyPredictor
from ..models.difficulty_calibrator import DifficultyCalibrator
from ..models.question_recommender import QuestionRecommender
from ..models.learning_path import LearningPathGenerator
from ..models.gap_analyzer import GapAnalyzer
from ..models.misconception_detector import MisconceptionDetector
from ..models.misconception_analyzer import MisconceptionAnalyzer
from ..knowledge_graph.construction import KnowledgeGraphConstructor
from ..knowledge_graph.query import KnowledgeGraphQuery
from ..training.pipeline import TrainingPipeline
from ..training.evaluator import ModelEvaluator
from ..export.model_exporter import ModelExporter

# Try to import OpenAI client
try:
    from ..openai.client import OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIClient = None

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manage service initialization, dependencies, and lifecycle."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize service manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = OrchestratorConfig(config_path)
        
        # Service registry
        self.services: Dict[str, Any] = {}
        self.initialized = False
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.featured_data: Optional[pd.DataFrame] = None
        self.knowledge_graph = None
    
    def initialize_services(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize all services in dependency order.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            Dictionary of initialized services
        """
        if self.initialized:
            logger.warning("Services already initialized. Skipping re-initialization.")
            return self.services
        
        logger.info("Initializing services...")
        
        # Get initialization order
        init_order = self.config.get_initialization_order()
        
        # Initialize services in order
        for service_name in init_order:
            try:
                logger.info(f"Initializing {service_name}...")
                self._initialize_service(service_name, data_path)
            except Exception as e:
                logger.error(f"Failed to initialize {service_name}: {e}")
                raise
        
        self.initialized = True
        logger.info("All services initialized successfully!")
        
        return self.services
    
    def _initialize_service(self, service_name: str, data_path: Optional[str] = None):
        """
        Initialize a specific service.
        
        Args:
            service_name: Name of the service to initialize
            data_path: Optional path to data file
        """
        if service_name == 'openai_client':
            # Initialize OpenAI client if available
            if OPENAI_AVAILABLE:
                try:
                    with open(self.config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    openai_config = config.get('openai', {})
                    
                    openai_client = OpenAIClient(
                        api_url=openai_config.get('api_url', 'http://192.168.0.4:1601'),
                        model=openai_config.get('model', 'qwen/qwen3-4b-thinking-2507'),
                        timeout=openai_config.get('timeout', 30),
                        max_retries=openai_config.get('max_retries', 3)
                    )
                    
                    if openai_client.is_available():
                        self.services['openai_client'] = openai_client
                        logger.info("OpenAI client initialized successfully")
                    else:
                        logger.info("OpenAI client not available, services will use local processing")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {e}")
            else:
                logger.info("OpenAI module not available")
        
        elif service_name == 'data_loader':
            if data_path is None:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                data_path = config.get('data', {}).get('input_file', 'question-response.json')
            self.data = load_dataframe(data_path)
            self.services['data_loader'] = {'data': self.data}
            logger.info(f"Loaded {len(self.data)} records")
        
        elif service_name == 'data_preprocessor':
            self.services['data_preprocessor'] = DataPreprocessor(self.config_path)
            if self.data is not None:
                self.processed_data = self.services['data_preprocessor'].preprocess(self.data)
                logger.info(f"Preprocessed {len(self.processed_data)} records")
        
        elif service_name == 'feature_engineer':
            # Feature engineer may need knowledge graph later, but initialize now
            self.services['feature_engineer'] = FeatureEngineer(
                self.config_path,
                graph=None,  # Will be set later if needed
                graph_embeddings=None
            )
            if self.processed_data is not None:
                self.featured_data = self.services['feature_engineer'].engineer_features(
                    self.processed_data,
                    include_text_embeddings=True,
                    include_graph_features=False  # Graph not ready yet
                )
                logger.info("Feature engineering completed")
        
        elif service_name == 'misconception_detector':
            self.services['misconception_detector'] = MisconceptionDetector(self.config_path)
            if self.processed_data is not None:
                # Detect misconceptions
                df_with_misconceptions = self.services['misconception_detector'].detect_misconceptions_from_data(
                    self.processed_data
                )
                misconceptions = self.services['misconception_detector'].catalog_misconceptions(
                    df_with_misconceptions
                )
                logger.info(f"Detected {len(misconceptions)} misconceptions")
        
        elif service_name == 'knowledge_graph_constructor':
            preprocessor = self.get_service('data_preprocessor')
            misconception_detector = self.get_service('misconception_detector')
            
            self.services['knowledge_graph_constructor'] = KnowledgeGraphConstructor(self.config_path)
            if self.processed_data is not None:
                self.knowledge_graph = self.services['knowledge_graph_constructor'].construct(
                    self.processed_data,
                    misconception_detector
                )
                logger.info(f"Knowledge graph built with {len(self.knowledge_graph.nodes())} nodes")
        
        elif service_name == 'knowledge_graph_query':
            if self.knowledge_graph is None:
                # Build graph if not already built
                kg_constructor = self.get_service('knowledge_graph_constructor')
                if kg_constructor and self.processed_data is not None:
                    misconception_detector = self.get_service('misconception_detector')
                    self.knowledge_graph = kg_constructor.construct(
                        self.processed_data,
                        misconception_detector
                    )
            
            if self.knowledge_graph is not None:
                self.services['knowledge_graph_query'] = KnowledgeGraphQuery(self.knowledge_graph)
                logger.info("Knowledge graph query service initialized")
            else:
                logger.warning("Knowledge graph not available, initializing query service without graph")
                self.services['knowledge_graph_query'] = KnowledgeGraphQuery(None)
        
        elif service_name == 'competency_predictor':
            self.services['competency_predictor'] = CompetencyPredictor()
            logger.info("Competency predictor initialized")
        
        elif service_name == 'difficulty_calibrator':
            self.services['difficulty_calibrator'] = DifficultyCalibrator()
            logger.info("Difficulty calibrator initialized")
        
        elif service_name == 'question_recommender':
            competency_predictor = self.get_service('competency_predictor')
            difficulty_calibrator = self.get_service('difficulty_calibrator')
            knowledge_graph_query = self.get_service('knowledge_graph_query')
            
            self.services['question_recommender'] = QuestionRecommender(
                competency_predictor,
                difficulty_calibrator,
                knowledge_graph_query,
                self.config_path
            )
            
            # Set question pool if data is available
            if self.processed_data is not None:
                self.services['question_recommender'].set_question_pool(self.processed_data)
                logger.info("Question recommender initialized with question pool")
            else:
                logger.info("Question recommender initialized (no question pool)")
        
        elif service_name == 'learning_path_generator':
            competency_predictor = self.get_service('competency_predictor')
            knowledge_graph_query = self.get_service('knowledge_graph_query')
            
            self.services['learning_path_generator'] = LearningPathGenerator(
                competency_predictor,
                knowledge_graph_query,
                self.config_path
            )
            logger.info("Learning path generator initialized")
        
        elif service_name == 'gap_analyzer':
            competency_predictor = self.get_service('competency_predictor')
            knowledge_graph_query = self.get_service('knowledge_graph_query')
            
            self.services['gap_analyzer'] = GapAnalyzer(
                competency_predictor,
                knowledge_graph_query,
                self.config_path
            )
            logger.info("Gap analyzer initialized")
        
        elif service_name == 'misconception_analyzer':
            misconception_detector = self.get_service('misconception_detector')
            knowledge_graph_query = self.get_service('knowledge_graph_query')
            
            self.services['misconception_analyzer'] = MisconceptionAnalyzer(
                misconception_detector,
                knowledge_graph_query,
                self.config_path
            )
            logger.info("Misconception analyzer initialized")
        
        elif service_name == 'training_pipeline':
            self.services['training_pipeline'] = TrainingPipeline(self.config_path)
            logger.info("Training pipeline initialized")
        
        elif service_name == 'model_evaluator':
            self.services['model_evaluator'] = ModelEvaluator()
            logger.info("Model evaluator initialized")
        
        elif service_name == 'model_exporter':
            self.services['model_exporter'] = ModelExporter(self.config_path)
            logger.info("Model exporter initialized")
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service by name.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service instance
        """
        if not self.initialized:
            raise RuntimeError("Services not initialized. Call initialize_services() first.")
        
        if service_name not in self.services:
            raise ValueError(f"Service '{service_name}' not found. Available services: {list(self.services.keys())}")
        
        return self.services[service_name]
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all services.
        
        Returns:
            Dictionary with health status of each service
        """
        health_status = {
            'overall': 'healthy',
            'services': {}
        }
        
        for service_name, service in self.services.items():
            try:
                # Basic health check - service exists and is not None
                if service is None:
                    health_status['services'][service_name] = 'unhealthy'
                    health_status['overall'] = 'degraded'
                else:
                    health_status['services'][service_name] = 'healthy'
            except Exception as e:
                health_status['services'][service_name] = f'unhealthy: {str(e)}'
                health_status['overall'] = 'degraded'
        
        return health_status
    
    def shutdown(self):
        """Gracefully shutdown all services."""
        logger.info("Shutting down services...")
        
        # Clear service references
        self.services.clear()
        self.initialized = False
        
        # Clear data
        self.data = None
        self.processed_data = None
        self.featured_data = None
        self.knowledge_graph = None
        
        logger.info("Services shut down successfully")
    
    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all initialized services.
        
        Returns:
            Dictionary of all services
        """
        return self.services.copy()
    
    def is_initialized(self) -> bool:
        """Check if services are initialized."""
        return self.initialized
