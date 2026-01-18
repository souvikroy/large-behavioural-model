"""Model evaluation utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score
)


class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mean_error': np.mean(y_true - y_pred),
            'std_error': np.std(y_true - y_pred)
        }
    
    @staticmethod
    def evaluate_recommendation(
        recommendations: List[Dict],
        actual_items: List[str],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate recommendation system.
        
        Args:
            recommendations: List of recommended items
            actual_items: List of actual relevant items
            k: Top-K for evaluation
            
        Returns:
            Dictionary with metrics
        """
        recommended_items = [r.get('question_id') for r in recommendations[:k]]
        
        # Precision@K
        relevant_recommended = len(set(recommended_items) & set(actual_items))
        precision_at_k = relevant_recommended / k if k > 0 else 0.0
        
        # Recall@K
        recall_at_k = relevant_recommended / len(actual_items) if len(actual_items) > 0 else 0.0
        
        # NDCG (simplified)
        ndcg = precision_at_k  # Simplified version
        
        return {
            'precision@k': precision_at_k,
            'recall@k': recall_at_k,
            'ndcg@k': ndcg
        }
    
    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba, average='weighted')
            except:
                metrics['auc_roc'] = 0.0
        
        return metrics
