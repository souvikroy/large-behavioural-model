"""Question difficulty calibration model."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import yaml
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class DifficultyCalibrator:
    """Calibrate and predict question difficulty (3PL scores)."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize difficulty calibrator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config.get('models', {}).get('difficulty_calibrator', {})
        self.algorithm = self.model_config.get('algorithm', 'random_forest')
        
        self.model = None
        self.feature_columns = None
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for training/prediction.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (features, target) arrays
        """
        # Select feature columns (exclude target and metadata)
        exclude_cols = [
            '3PL', 'question_text', 'learning_objective', 
            'learning_unit', 'question_id', 'competency_score',
            'competency_category', 'competency_score_range', 
            'Predicted_competency_score_range', '3PL_normalized'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Store feature columns
        if self.feature_columns is None:
            self.feature_columns = feature_cols
        
        # Get features
        X = df[feature_cols].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get target if available
        y = None
        if '3PL' in df.columns:
            y = pd.to_numeric(df['3PL'], errors='coerce').values
            y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train difficulty calibration model.
        
        Args:
            train_df: Training DataFrame
            val_df: Optional validation DataFrame
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features
        X_train, y_train = self.prepare_features(train_df)
        
        # Train model
        if self.algorithm == 'random_forest':
            n_estimators = self.model_config.get('n_estimators', 100)
            max_depth = self.model_config.get('max_depth', 10)
            
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Default to Random Forest
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred)
        }
        
        if val_df is not None:
            X_val, y_val = self.prepare_features(val_df)
            val_pred = self.model.predict(X_val)
            metrics.update({
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred)
            })
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict difficulty (3PL) scores.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of predicted 3PL scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, _ = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        return predictions
    
    def adjust_difficulty_for_prerequisites(
        self,
        predicted_difficulty: float,
        prerequisites_met: bool
    ) -> float:
        """
        Adjust difficulty based on prerequisite status.
        
        Args:
            predicted_difficulty: Predicted difficulty score
            prerequisites_met: Whether prerequisites are met
            
        Returns:
            Adjusted difficulty score
        """
        if not prerequisites_met:
            # Increase difficulty if prerequisites not met
            return predicted_difficulty + 1.0
        return predicted_difficulty
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'algorithm': self.algorithm
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.algorithm = model_data['algorithm']
