"""Data preprocessing module."""

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import yaml
from pathlib import Path


class DataPreprocessor:
    """Preprocess question-response data for modeling."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.data_config = self.config.get('data', {})
        
        # Initialize encoders and scalers
        self.label_encoders = {}
        self.scaler_3pl = StandardScaler()
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file (uses config if None)
            
        Returns:
            DataFrame with loaded data
        """
        if file_path is None:
            file_path = self.config['data']['input_file']
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        strategy = self.preprocessing_config.get('handle_missing', 'drop')
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'fill':
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else '')
        
        return df
    
    def normalize_3pl_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize 3PL scores.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized 3PL scores
        """
        if '3PL' in df.columns:
            # Convert to numeric
            df['3PL'] = pd.to_numeric(df['3PL'], errors='coerce')
            
            if self.preprocessing_config.get('normalize_3pl', True):
                # Standardize 3PL scores
                df['3PL_normalized'] = self.scaler_3pl.fit_transform(df[['3PL']])
            else:
                df['3PL_normalized'] = df['3PL']
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        categorical_cols = ['Subject', 'grade', 'chapter', 'Bloom_tag']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        return df
    
    def create_competency_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create competency level categories.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with competency categories
        """
        if 'competency_score' in df.columns:
            df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
            
            def categorize_competency(score):
                if pd.isna(score):
                    return 'Unknown'
                elif score < 0.3:
                    return 'Low'
                elif score < 0.7:
                    return 'Medium'
                else:
                    return 'High'
            
            df['competency_category'] = df['competency_score'].apply(categorize_competency)
        
        return df
    
    def create_difficulty_bins(self, df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
        """
        Create difficulty bins from 3PL scores.
        
        Args:
            df: Input DataFrame
            n_bins: Number of bins
            
        Returns:
            DataFrame with difficulty bins
        """
        if '3PL' in df.columns:
            df['3PL'] = pd.to_numeric(df['3PL'], errors='coerce')
            df['difficulty_bin'] = pd.qcut(
                df['3PL'], 
                q=n_bins, 
                labels=[f'Difficulty_{i+1}' for i in range(n_bins)],
                duplicates='drop'
            )
        
        return df
    
    def extract_question_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic features from question text.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with question text features
        """
        if 'question_text' in df.columns:
            # Text length
            df['question_text_length'] = df['question_text'].astype(str).apply(len)
            
            # Word count
            df['question_word_count'] = df['question_text'].astype(str).apply(
                lambda x: len(x.split())
            )
            
            # Check for misconception indicators
            misconception_keywords = self.config.get('nlp', {}).get('misconception_keywords', [])
            for keyword in misconception_keywords:
                df[f'contains_{keyword}'] = df['question_text'].astype(str).str.contains(
                    keyword, case=False, na=False
                ).astype(int)
        
        return df
    
    def identify_question_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify patterns in questions (multiple attempts, variance).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with pattern indicators
        """
        # Convert numeric columns
        if 'competency_score' in df.columns:
            df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
        
        if 'total_attempt' in df.columns:
            df['total_attempt'] = pd.to_numeric(df['total_attempt'], errors='coerce')
        
        # Calculate variance in competency scores for same question
        if 'question_id' in df.columns:
            question_stats = df.groupby('question_id')['competency_score'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            question_stats.columns = ['question_id', 'mean_competency', 'competency_std', 'attempt_count']
            
            df = df.merge(question_stats, on='question_id', how='left')
            df['high_variance'] = (df['competency_std'] > df['competency_std'].quantile(0.75)).astype(int)
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Normalize 3PL scores
        df = self.normalize_3pl_scores(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Create competency categories
        df = self.create_competency_categories(df)
        
        # Create difficulty bins
        df = self.create_difficulty_bins(df)
        
        # Extract question text features
        df = self.extract_question_text_features(df)
        
        # Identify question patterns
        df = self.identify_question_patterns(df)
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'competency_score'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_ratio = self.data_config.get('train_split', 0.7)
        val_ratio = self.data_config.get('val_split', 0.15)
        test_ratio = self.data_config.get('test_split', 0.15)
        
        random_seed = self.data_config.get('random_seed', 42)
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - train_ratio), 
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=random_seed,
            shuffle=True
        )
        
        return train_df, val_df, test_df
    
    def get_feature_columns(self) -> Dict[str, list]:
        """
        Get lists of feature columns by type.
        
        Returns:
            Dictionary with feature column lists
        """
        return {
            'categorical': ['Subject_encoded', 'grade_encoded', 'chapter_encoded', 'Bloom_tag_encoded'],
            'numeric': ['3PL', '3PL_normalized', 'total_attempt', 'question_text_length', 'question_word_count'],
            'text': ['question_text', 'learning_objective', 'learning_unit'],
            'target': ['competency_score', 'competency_category']
        }
