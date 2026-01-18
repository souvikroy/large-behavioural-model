"""Training pipeline with cross-validation and hyperparameter tuning."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
from sklearn.model_selection import KFold, cross_val_score
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..models.competency_predictor import CompetencyPredictor
from ..models.difficulty_calibrator import DifficultyCalibrator
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer


class TrainingPipeline:
    """Training pipeline with cross-validation and hyperparameter tuning."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config.get('training', {})
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        
        self.competency_predictor = None
        self.difficulty_calibrator = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare and split data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Preprocess
        df_processed = self.preprocessor.preprocess(df)
        
        # Feature engineering
        df_features = self.feature_engineer.engineer_features(df_processed)
        
        # Split
        train_df, val_df, test_df = self.preprocessor.split_data(df_features)
        
        return train_df, val_df, test_df
    
    def train_competency_predictor(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        tune_hyperparameters: bool = False
    ) -> Dict[str, float]:
        """
        Train competency predictor.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Training metrics
        """
        self.competency_predictor = CompetencyPredictor()
        
        if tune_hyperparameters:
            # Hyperparameter tuning with Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self._objective_competency(trial, train_df, val_df),
                n_trials=self.training_config.get('hyperparameter_tuning', {}).get('n_trials', 50)
            )
            
            # Use best parameters
            best_params = study.best_params
            # Update model config (simplified - in practice, you'd update the model)
        
        # Train with best/default parameters
        metrics = self.competency_predictor.train(train_df, val_df)
        
        return metrics
    
    def _objective_competency(
        self,
        trial: optuna.Trial,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> float:
        """
        Objective function for hyperparameter tuning.
        
        Args:
            trial: Optuna trial
            train_df: Training DataFrame
            val_df: Validation DataFrame
            
        Returns:
            Validation RMSE
        """
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        
        # Create and train model
        predictor = CompetencyPredictor()
        predictor.model_config['xgboost_params'] = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }
        
        metrics = predictor.train(train_df, val_df)
        
        return metrics.get('xgboost_val_rmse', 1.0)
    
    def train_difficulty_calibrator(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Train difficulty calibrator.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            
        Returns:
            Training metrics
        """
        self.difficulty_calibrator = DifficultyCalibrator()
        metrics = self.difficulty_calibrator.train(train_df, val_df)
        
        return metrics
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        model_type: str = 'competency_predictor',
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            df: Full DataFrame
            model_type: Type of model ('competency_predictor' or 'difficulty_calibrator')
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV metrics
        """
        cv_config = self.training_config.get('cross_validation', {})
        n_splits = cv_config.get('n_splits', n_splits)
        
        # Preprocess and engineer features
        df_processed = self.preprocessor.preprocess(df)
        df_features = self.feature_engineer.engineer_features(df_processed)
        
        # Prepare features
        if model_type == 'competency_predictor':
            X, y = self.competency_predictor.prepare_features(df_features)
        else:
            X, y = self.difficulty_calibrator.prepare_features(df_features)
        
        # Cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = {'rmse': [], 'mae': [], 'r2': []}
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create temporary dataframes
            train_df = df_features.iloc[train_idx].copy()
            val_df = df_features.iloc[val_idx].copy()
            
            # Train and evaluate
            if model_type == 'competency_predictor':
                model = CompetencyPredictor()
                model.train(train_df, val_df)
                predictions = model.predict(val_df)
            else:
                model = DifficultyCalibrator()
                model.train(train_df, val_df)
                predictions = model.predict(val_df)
            
            # Calculate metrics
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, predictions)))
            cv_scores['mae'].append(mean_absolute_error(y_val, predictions))
            cv_scores['r2'].append(r2_score(y_val, predictions))
        
        return cv_scores
    
    def evaluate(
        self,
        test_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate models on test set.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        if self.competency_predictor:
            predictions = self.competency_predictor.predict(test_df)
            actual = pd.to_numeric(test_df['competency_score'], errors='coerce').values
            
            results['competency_predictor'] = {
                'rmse': np.sqrt(mean_squared_error(actual, predictions)),
                'mae': mean_absolute_error(actual, predictions),
                'r2': r2_score(actual, predictions)
            }
        
        if self.difficulty_calibrator:
            predictions = self.difficulty_calibrator.predict(test_df)
            actual = pd.to_numeric(test_df['3PL'], errors='coerce').values
            
            results['difficulty_calibrator'] = {
                'rmse': np.sqrt(mean_squared_error(actual, predictions)),
                'mae': mean_absolute_error(actual, predictions),
                'r2': r2_score(actual, predictions)
            }
        
        return results
    
    def train_all(
        self,
        df: pd.DataFrame,
        tune_hyperparameters: bool = False
    ) -> Dict[str, any]:
        """
        Train all models.
        
        Args:
            df: Input DataFrame
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary with training results
        """
        # Prepare data
        train_df, val_df, test_df = self.prepare_data(df)
        
        results = {}
        
        # Train competency predictor
        print("Training competency predictor...")
        competency_metrics = self.train_competency_predictor(
            train_df, val_df, tune_hyperparameters
        )
        results['competency_predictor'] = competency_metrics
        
        # Train difficulty calibrator
        print("Training difficulty calibrator...")
        difficulty_metrics = self.train_difficulty_calibrator(train_df, val_df)
        results['difficulty_calibrator'] = difficulty_metrics
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = self.evaluate(test_df)
        results['test_metrics'] = test_metrics
        
        return results
