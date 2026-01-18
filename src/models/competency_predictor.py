"""Competency prediction model."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import yaml
import joblib

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception):
    LIGHTGBM_AVAILABLE = False
    LGBMRegressor = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, Exception):
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class CompetencyPredictor:
    """Predict student competency scores for questions."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize competency predictor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config.get('models', {}).get('competency_predictor', {})
        self.algorithm = self.model_config.get('algorithm', 'ensemble')
        
        self.xgboost_model = None
        self.lightgbm_model = None
        self.neural_network_model = None
        
        self.feature_columns = None
    
    def _build_neural_network(self, input_dim: int) -> nn.Module:
        """
        Build neural network model.
        
        Args:
            input_dim: Input feature dimension
            
        Returns:
            Neural network model
        """
        nn_config = self.model_config.get('neural_network', {})
        hidden_layers = nn_config.get('hidden_layers', [256, 128, 64])
        dropout = nn_config.get('dropout', 0.3)
        activation = nn_config.get('activation', 'relu')
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        
        return nn.Sequential(*layers)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for training/prediction.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (features, target) arrays
        """
        # Select feature columns (exclude target and metadata)
        exclude_cols = [
            'competency_score', 'question_text', 'learning_objective', 
            'learning_unit', 'question_id', 'competency_category',
            'competency_score_range', 'Predicted_competency_score_range'
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
        if 'competency_score' in df.columns:
            y = pd.to_numeric(df['competency_score'], errors='coerce').values
            y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train competency prediction model.
        
        Args:
            train_df: Training DataFrame
            val_df: Optional validation DataFrame
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features
        X_train, y_train = self.prepare_features(train_df)
        
        metrics = {}
        
        if self.algorithm in ['xgboost', 'ensemble'] and XGBOOST_AVAILABLE:
            # Train XGBoost
            xgb_params = self.model_config.get('xgboost_params', {})
            self.xgboost_model = XGBRegressor(
                n_estimators=xgb_params.get('n_estimators', 100),
                max_depth=xgb_params.get('max_depth', 6),
                learning_rate=xgb_params.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1
            )
            self.xgboost_model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = self.xgboost_model.predict(X_train)
            metrics['xgboost_train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
            metrics['xgboost_train_mae'] = mean_absolute_error(y_train, train_pred)
            metrics['xgboost_train_r2'] = r2_score(y_train, train_pred)
            
            if val_df is not None:
                X_val, y_val = self.prepare_features(val_df)
                val_pred = self.xgboost_model.predict(X_val)
                metrics['xgboost_val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
                metrics['xgboost_val_mae'] = mean_absolute_error(y_val, val_pred)
                metrics['xgboost_val_r2'] = r2_score(y_val, val_pred)
        
        if self.algorithm in ['lightgbm', 'ensemble'] and LIGHTGBM_AVAILABLE:
            # Train LightGBM
            lgbm_params = self.model_config.get('lightgbm_params', {})
            self.lightgbm_model = LGBMRegressor(
                n_estimators=lgbm_params.get('n_estimators', 100),
                max_depth=lgbm_params.get('max_depth', 6),
                learning_rate=lgbm_params.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.lightgbm_model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = self.lightgbm_model.predict(X_train)
            metrics['lightgbm_train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
            metrics['lightgbm_train_mae'] = mean_absolute_error(y_train, train_pred)
            metrics['lightgbm_train_r2'] = r2_score(y_train, train_pred)
            
            if val_df is not None:
                X_val, y_val = self.prepare_features(val_df)
                val_pred = self.lightgbm_model.predict(X_val)
                metrics['lightgbm_val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
                metrics['lightgbm_val_mae'] = mean_absolute_error(y_val, val_pred)
                metrics['lightgbm_val_r2'] = r2_score(y_val, val_pred)
        
        if self.algorithm in ['neural_network', 'ensemble'] and TORCH_AVAILABLE:
            # Train Neural Network
            self.neural_network_model = self._build_neural_network(X_train.shape[1])
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.neural_network_model.parameters(), lr=0.001)
            
            # Train
            epochs = 50
            batch_size = 32
            
            for epoch in range(epochs):
                self.neural_network_model.train()
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.neural_network_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            self.neural_network_model.eval()
            with torch.no_grad():
                train_pred_tensor = self.neural_network_model(X_train_tensor)
                train_pred = train_pred_tensor.squeeze().numpy()
            
            metrics['neural_network_train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
            metrics['neural_network_train_mae'] = mean_absolute_error(y_train, train_pred)
            metrics['neural_network_train_r2'] = r2_score(y_train, train_pred)
            
            if val_df is not None:
                X_val, y_val = self.prepare_features(val_df)
                X_val_tensor = torch.FloatTensor(X_val)
                
                with torch.no_grad():
                    val_pred_tensor = self.neural_network_model(X_val_tensor)
                    val_pred = val_pred_tensor.squeeze().numpy()
                
                metrics['neural_network_val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
                metrics['neural_network_val_mae'] = mean_absolute_error(y_val, val_pred)
                metrics['neural_network_val_r2'] = r2_score(y_val, val_pred)
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict competency scores.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of predicted competency scores
        """
        X, _ = self.prepare_features(df)
        
        predictions = []
        
        if self.algorithm == 'ensemble':
            # Ensemble prediction
            preds = []
            
            if self.xgboost_model:
                preds.append(self.xgboost_model.predict(X))
            if self.lightgbm_model:
                preds.append(self.lightgbm_model.predict(X))
            if self.neural_network_model:
                self.neural_network_model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    nn_pred = self.neural_network_model(X_tensor).squeeze().numpy()
                    preds.append(nn_pred)
            
            if preds:
                predictions = np.mean(preds, axis=0)
            else:
                predictions = np.zeros(len(X))
        elif self.algorithm == 'xgboost' and self.xgboost_model:
            predictions = self.xgboost_model.predict(X)
        elif self.algorithm == 'lightgbm' and self.lightgbm_model:
            predictions = self.lightgbm_model.predict(X)
        elif self.algorithm == 'neural_network' and self.neural_network_model:
            self.neural_network_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = self.neural_network_model(X_tensor).squeeze().numpy()
        else:
            predictions = np.zeros(len(X))
        
        # Clip predictions to [0, 1]
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'algorithm': self.algorithm,
            'feature_columns': self.feature_columns,
            'xgboost_model': self.xgboost_model,
            'lightgbm_model': self.lightgbm_model,
            'neural_network_state': self.neural_network_model.state_dict() if self.neural_network_model else None,
            'neural_network_config': self.model_config.get('neural_network', {}) if self.neural_network_model else None
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        self.algorithm = model_data['algorithm']
        self.feature_columns = model_data['feature_columns']
        self.xgboost_model = model_data['xgboost_model']
        self.lightgbm_model = model_data['lightgbm_model']
        
        if model_data['neural_network_state']:
            input_dim = len(model_data['feature_columns'])
            self.neural_network_model = self._build_neural_network(input_dim)
            self.neural_network_model.load_state_dict(model_data['neural_network_state'])
