"""Training orchestrator for coordinating training workflows."""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from .service_manager import ServiceManager

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Coordinate training workflows."""
    
    def __init__(self, service_manager: ServiceManager):
        """
        Initialize training orchestrator.
        
        Args:
            service_manager: ServiceManager instance
        """
        self.service_manager = service_manager
        self.training_results = {}
    
    def run_training_pipeline(
        self,
        data_path: Optional[str] = None,
        tune_hyperparameters: bool = False,
        export_models: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full training pipeline.
        
        Args:
            data_path: Optional path to data file
            tune_hyperparameters: Whether to tune hyperparameters
            export_models: Whether to export trained models
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training pipeline...")
        
        try:
            # Step 1: Prepare data
            train_df, val_df, test_df = self.prepare_data(data_path)
            
            # Step 2: Train models
            training_metrics = self.train_models(
                train_df,
                val_df,
                tune_hyperparameters
            )
            
            # Step 3: Evaluate models
            evaluation_metrics = self.evaluate_models(test_df)
            
            # Step 4: Export models if requested
            export_paths = {}
            if export_models:
                export_paths = self.export_models(training_metrics, evaluation_metrics)
            
            self.training_results = {
                'training_metrics': training_metrics,
                'evaluation_metrics': evaluation_metrics,
                'export_paths': export_paths,
                'status': 'success'
            }
            
            logger.info("Training pipeline completed successfully!")
            return self.training_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            self.training_results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def prepare_data(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Coordinate data loading and preprocessing.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Preparing data...")
        
        # Get services
        preprocessor = self.service_manager.get_service('data_preprocessor')
        feature_engineer = self.service_manager.get_service('feature_engineer')
        
        # Load data if not already loaded
        if self.service_manager.data is None:
            if data_path is None:
                import yaml
                with open(self.service_manager.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                data_path = config.get('data', {}).get('input_file', 'question-response.json')
            
            from ..data.loader import load_dataframe
            self.service_manager.data = load_dataframe(data_path)
        
        # Preprocess
        df_processed = preprocessor.preprocess(self.service_manager.data)
        self.service_manager.processed_data = df_processed
        
        # Engineer features
        # Update feature engineer with knowledge graph if available
        if self.service_manager.knowledge_graph is not None:
            feature_engineer.graph = self.service_manager.knowledge_graph
            feature_engineer.graph_query = self.service_manager.get_service('knowledge_graph_query')
        
        df_features = feature_engineer.engineer_features(
            df_processed,
            include_text_embeddings=True,
            include_graph_features=(self.service_manager.knowledge_graph is not None)
        )
        self.service_manager.featured_data = df_features
        
        # Split data
        train_df, val_df, test_df = preprocessor.split_data(df_features)
        
        logger.info(f"Data prepared: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def train_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        tune_hyperparameters: bool = False
    ) -> Dict[str, Any]:
        """
        Coordinate model training.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training models...")
        
        training_pipeline = self.service_manager.get_service('training_pipeline')
        
        # Train all models using the training pipeline
        results = training_pipeline.train_all(
            pd.concat([train_df, val_df], ignore_index=True),
            tune_hyperparameters=tune_hyperparameters
        )
        
        logger.info("Model training completed")
        return results
    
    def evaluate_models(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Coordinate model evaluation.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating models...")
        
        training_pipeline = self.service_manager.get_service('training_pipeline')
        model_evaluator = self.service_manager.get_service('model_evaluator')
        
        # Evaluate using training pipeline
        evaluation_results = training_pipeline.evaluate(test_df)
        
        logger.info("Model evaluation completed")
        return evaluation_results
    
    def export_models(
        self,
        training_metrics: Dict[str, Any],
        evaluation_metrics: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        """
        Coordinate model export.
        
        Args:
            training_metrics: Training metrics
            evaluation_metrics: Evaluation metrics
            
        Returns:
            Dictionary with export paths
        """
        logger.info("Exporting models...")
        
        model_exporter = self.service_manager.get_service('model_exporter')
        
        # Prepare models dictionary
        models_to_export = {}
        model_metrics = {}
        
        # Get trained models from training pipeline
        training_pipeline = self.service_manager.get_service('training_pipeline')
        
        if training_pipeline.competency_predictor:
            models_to_export['competency_predictor'] = training_pipeline.competency_predictor
            if 'competency_predictor' in evaluation_metrics:
                model_metrics['competency_predictor'] = evaluation_metrics['competency_predictor']
        
        if training_pipeline.difficulty_calibrator:
            models_to_export['difficulty_calibrator'] = training_pipeline.difficulty_calibrator
            if 'difficulty_calibrator' in evaluation_metrics:
                model_metrics['difficulty_calibrator'] = evaluation_metrics['difficulty_calibrator']
        
        # Export knowledge graph if available
        knowledge_graph = self.service_manager.knowledge_graph
        
        # Export all models
        export_paths = model_exporter.export_all(
            models=models_to_export,
            knowledge_graph=knowledge_graph,
            model_metrics=model_metrics
        )
        
        logger.info("Model export completed")
        return export_paths
    
    def cross_validate(
        self,
        data_path: Optional[str] = None,
        model_type: str = 'competency_predictor',
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            data_path: Optional path to data file
            model_type: Type of model to validate
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing cross-validation for {model_type}...")
        
        # Prepare data
        if self.service_manager.featured_data is None:
            self.prepare_data(data_path)
        
        training_pipeline = self.service_manager.get_service('training_pipeline')
        
        # Perform cross-validation
        cv_results = training_pipeline.cross_validate(
            self.service_manager.featured_data,
            model_type=model_type,
            n_splits=n_splits
        )
        
        logger.info("Cross-validation completed")
        return cv_results
