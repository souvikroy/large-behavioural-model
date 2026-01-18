"""Feature engineering pipeline."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Optional
import yaml
import networkx as nx

from ..nlp.text_embeddings import TextEmbedder
from ..knowledge_graph.embeddings import GraphEmbedder
from ..knowledge_graph.query import KnowledgeGraphQuery


class FeatureEngineer:
    """Engineer features for machine learning models."""
    
    def __init__(self, config_path: str = "config.yaml", graph=None, graph_embeddings=None):
        """
        Initialize feature engineer.
        
        Args:
            config_path: Path to configuration file
            graph: Optional knowledge graph
            graph_embeddings: Optional graph embeddings dictionary
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.nlp_config = self.config.get('nlp', {})
        self.text_embedder = TextEmbedder(
            model_name=self.nlp_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        
        self.graph = graph
        self.graph_embeddings = graph_embeddings
        self.graph_query = KnowledgeGraphQuery(graph) if graph else None
        
        # Encoders
        self.onehot_encoders = {}
        self.scalers = {}
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create one-hot encoded categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with one-hot encoded features
        """
        df = df.copy()
        categorical_cols = ['Subject', 'grade', 'Bloom_tag']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.onehot_encoders:
                    self.onehot_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = self.onehot_encoders[col].fit_transform(df[[col]])
                else:
                    encoded = self.onehot_encoders[col].transform(df[[col]])
                
                # Create column names
                feature_names = [f'{col}_{cat}' for cat in self.onehot_encoders[col].categories_[0]]
                
                # Add to dataframe
                for i, feature_name in enumerate(feature_names):
                    df[feature_name] = encoded[:, i]
        
        return df
    
    def create_text_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create text embeddings for LO, LU, and question text.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with embedding features
        """
        df = df.copy()
        
        # Embed learning objectives
        if 'learning_objective' in df.columns:
            lo_texts = df['learning_objective'].astype(str).tolist()
            lo_embeddings = self.text_embedder.embed(lo_texts)
            
            # Add embedding dimensions as columns
            for i in range(lo_embeddings.shape[1]):
                df[f'lo_embedding_{i}'] = lo_embeddings[:, i]
        
        # Embed learning units
        if 'learning_unit' in df.columns:
            lu_texts = df['learning_unit'].astype(str).tolist()
            lu_embeddings = self.text_embedder.embed(lu_texts)
            
            for i in range(lu_embeddings.shape[1]):
                df[f'lu_embedding_{i}'] = lu_embeddings[:, i]
        
        # Embed question text
        if 'question_text' in df.columns:
            question_texts = df['question_text'].astype(str).tolist()
            question_embeddings = self.text_embedder.embed(question_texts)
            
            for i in range(question_embeddings.shape[1]):
                df[f'question_embedding_{i}'] = question_embeddings[:, i]
        
        return df
    
    def create_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from knowledge graph.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with graph features
        """
        df = df.copy()
        
        if self.graph is None or self.graph_query is None:
            return df
        
        # Graph-based features for each question (batch processed)
        graph_features = []
        
        # Batch collect question IDs
        question_ids = df['question_id'].astype(str).tolist() if 'question_id' in df.columns else []
        question_nodes = [f"question_{qid}" for qid in question_ids]
        
        # Pre-compute centrality if graph is small enough
        centrality_dict = {}
        try:
            if len(self.graph) < 10000:
                centrality = nx.degree_centrality(self.graph)
                centrality_dict = centrality
        except:
            pass
        
        # Process all questions
        for question_node in question_nodes:
            features = {}
            
            if question_node in self.graph:
                # Get node embeddings if available
                if self.graph_embeddings and question_node in self.graph_embeddings:
                    embedding = self.graph_embeddings[question_node]
                    for i, val in enumerate(embedding):
                        features[f'graph_embedding_{i}'] = val
                
                # Get connected concepts
                concepts = [
                    v for u, v, data in self.graph.edges(question_node, data=True)
                    if data.get('edge_type') == 'TESTS' and self.graph.nodes[v].get('node_type') == 'concept'
                ]
                
                features['num_concepts'] = len(concepts)
                
                # Get prerequisite distance
                if concepts:
                    # Find prerequisites for first concept
                    first_concept = concepts[0]
                    prerequisites = self.graph_query.find_prerequisites(first_concept)
                    features['num_prerequisites'] = len(prerequisites)
                else:
                    features['num_prerequisites'] = 0
                
                # Get misconception associations
                misconceptions = [
                    v for u, v, data in self.graph.edges(question_node, data=True)
                    if self.graph.nodes[v].get('node_type') == 'misconception'
                ]
                features['num_misconceptions'] = len(misconceptions)
                
                # Get centrality measure
                features['centrality'] = centrality_dict.get(question_node, 0.0)
            else:
                # Default values if node not in graph
                features['num_concepts'] = 0
                features['num_prerequisites'] = 0
                features['num_misconceptions'] = 0
                features['centrality'] = 0.0
            
            graph_features.append(features)
        
        # Add graph features to dataframe
        for feature_name in graph_features[0].keys() if graph_features else []:
            df[feature_name] = [f.get(feature_name, 0.0) for f in graph_features]
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features
        """
        df = df.copy()
        
        # Ensure numeric columns
        numeric_cols = ['3PL', '3PL_normalized', 'total_attempt', 'competency_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Interaction features
        if 'Subject_encoded' in df.columns and 'grade_encoded' in df.columns:
            df['subject_grade_interaction'] = df['Subject_encoded'] * df['grade_encoded']
        
        if 'chapter_encoded' in df.columns and 'Bloom_tag_encoded' in df.columns:
            df['chapter_bloom_interaction'] = df['chapter_encoded'] * df['Bloom_tag_encoded']
        
        # Difficulty-competency interaction
        if '3PL_normalized' in df.columns and 'competency_score' in df.columns:
            df['difficulty_competency_interaction'] = df['3PL_normalized'] * df['competency_score']
        
        # Misconception risk score (if available)
        if 'misconception_risk' in df.columns:
            df['misconception_risk_score'] = df['misconception_risk']
        else:
            # Calculate from indicators
            indicator_cols = [col for col in df.columns if col.startswith('contains_')]
            if indicator_cols:
                df['misconception_risk_score'] = df[indicator_cols].sum(axis=1) / len(indicator_cols)
            else:
                df['misconception_risk_score'] = 0.0
        
        # Normalize numeric features
        numeric_features = ['3PL_normalized', 'total_attempt', 'num_concepts', 'num_prerequisites']
        for feature in numeric_features:
            if feature in df.columns:
                if feature not in self.scalers:
                    self.scalers[feature] = StandardScaler()
                    df[feature] = self.scalers[feature].fit_transform(df[[feature]])
                else:
                    df[feature] = self.scalers[feature].transform(df[[feature]])
        
        return df
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        include_text_embeddings: bool = True,
        include_graph_features: bool = True
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            include_text_embeddings: Whether to include text embeddings
            include_graph_features: Whether to include graph features
            
        Returns:
            DataFrame with engineered features
        """
        # Categorical features
        df = self.create_categorical_features(df)
        
        # Text embeddings
        if include_text_embeddings:
            df = self.create_text_embeddings(df)
        
        # Graph features
        if include_graph_features:
            df = self.create_graph_features(df)
        
        # Derived features
        df = self.create_derived_features(df)
        
        return df
    
    def get_feature_columns(self) -> Dict[str, List[str]]:
        """
        Get lists of feature columns by type.
        
        Returns:
            Dictionary with feature column lists
        """
        return {
            'categorical': [col for col in self.onehot_encoders.keys()],
            'text_embeddings': [
                col for col in ['lo_embedding', 'lu_embedding', 'question_embedding']
                if any(c.startswith(col) for c in [])  # Would need actual column names
            ],
            'graph_features': ['num_concepts', 'num_prerequisites', 'num_misconceptions', 'centrality'],
            'derived': ['subject_grade_interaction', 'chapter_bloom_interaction', 'misconception_risk_score']
        }
