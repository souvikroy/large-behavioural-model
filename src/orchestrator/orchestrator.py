"""Main orchestrator combining service management and workflow coordination."""

import logging
from typing import Dict, Any, Optional

from .service_manager import ServiceManager
from .training_orchestrator import TrainingOrchestrator
from .runtime_orchestrator import RuntimeOrchestrator

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestrator for managing services and workflows."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.service_manager = ServiceManager(config_path)
        self.training_orchestrator = None
        self.runtime_orchestrator = None
        self.initialized = False
    
    def initialize(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize all services.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            Dictionary with initialization results
        """
        if self.initialized:
            logger.warning("Orchestrator already initialized")
            return {'status': 'already_initialized'}
        
        logger.info("Initializing orchestrator...")
        
        # Initialize services
        services = self.service_manager.initialize_services(data_path)
        
        # Initialize orchestrators
        self.training_orchestrator = TrainingOrchestrator(self.service_manager)
        self.runtime_orchestrator = RuntimeOrchestrator(self.service_manager)
        
        self.initialized = True
        
        # Health check
        health = self.service_manager.health_check()
        
        logger.info("Orchestrator initialized successfully!")
        
        return {
            'status': 'initialized',
            'services_initialized': len(services),
            'health': health
        }
    
    def train(
        self,
        data_path: Optional[str] = None,
        tune_hyperparameters: bool = False,
        export_models: bool = True
    ) -> Dict[str, Any]:
        """
        Run training pipeline.
        
        Args:
            data_path: Optional path to data file
            tune_hyperparameters: Whether to tune hyperparameters
            export_models: Whether to export trained models
            
        Returns:
            Training results
        """
        if not self.initialized:
            self.initialize(data_path)
        
        if self.training_orchestrator is None:
            self.training_orchestrator = TrainingOrchestrator(self.service_manager)
        
        return self.training_orchestrator.run_training_pipeline(
            data_path=data_path,
            tune_hyperparameters=tune_hyperparameters,
            export_models=export_models
        )
    
    def get_runtime_orchestrator(self) -> RuntimeOrchestrator:
        """
        Get runtime orchestrator for API use.
        
        Returns:
            RuntimeOrchestrator instance
        """
        if not self.initialized:
            self.initialize()
        
        if self.runtime_orchestrator is None:
            self.runtime_orchestrator = RuntimeOrchestrator(self.service_manager)
        
        return self.runtime_orchestrator
    
    def get_training_orchestrator(self) -> TrainingOrchestrator:
        """
        Get training orchestrator.
        
        Returns:
            TrainingOrchestrator instance
        """
        if not self.initialized:
            self.initialize()
        
        if self.training_orchestrator is None:
            self.training_orchestrator = TrainingOrchestrator(self.service_manager)
        
        return self.training_orchestrator
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service by name.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service instance
        """
        if not self.initialized:
            self.initialize()
        
        return self.service_manager.get_service(service_name)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all services.
        
        Returns:
            Health status
        """
        if not self.initialized:
            return {
                'overall': 'not_initialized',
                'services': {}
            }
        
        return self.service_manager.health_check()
    
    def shutdown(self):
        """Gracefully shutdown all services."""
        logger.info("Shutting down orchestrator...")
        
        if self.service_manager:
            self.service_manager.shutdown()
        
        self.training_orchestrator = None
        self.runtime_orchestrator = None
        self.initialized = False
        
        logger.info("Orchestrator shut down successfully")
    
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized."""
        return self.initialized
    
    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all initialized services.
        
        Returns:
            Dictionary of all services
        """
        if not self.initialized:
            return {}
        
        return self.service_manager.get_all_services()
