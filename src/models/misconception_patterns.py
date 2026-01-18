"""Misconception pattern identification and analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..nlp.question_analyzer import QuestionAnalyzer


class MisconceptionPatternIdentifier:
    """Identify and analyze misconception patterns."""
    
    def __init__(self, question_analyzer: QuestionAnalyzer):
        """
        Initialize pattern identifier.
        
        Args:
            question_analyzer: QuestionAnalyzer instance
        """
        self.question_analyzer = question_analyzer
    
    def identify_patterns_by_dimension(
        self, 
        df: pd.DataFrame,
        dimensions: List[str] = ['Subject', 'chapter', 'Bloom_tag']
    ) -> Dict[str, Dict]:
        """
        Identify misconception patterns by dimension.
        
        Args:
            df: DataFrame with questions and competency scores
            dimensions: List of dimensions to analyze
            
        Returns:
            Dictionary of patterns by dimension
        """
        df = df.copy()
        df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
        
        patterns = {}
        competency_threshold = 0.3
        
        for dimension in dimensions:
            if dimension not in df.columns:
                continue
            
            dimension_patterns = {}
            
            # Group by dimension value
            for value in df[dimension].unique():
                value_df = df[df[dimension] == value]
                low_comp_df = value_df[value_df['competency_score'] < competency_threshold]
                
                if len(low_comp_df) > 0:
                    dimension_patterns[value] = {
                        'total_questions': len(value_df),
                        'low_competency_count': len(low_comp_df),
                        'low_competency_rate': len(low_comp_df) / len(value_df),
                        'avg_competency': value_df['competency_score'].mean(),
                        'avg_low_competency': low_comp_df['competency_score'].mean()
                    }
            
            patterns[dimension] = dimension_patterns
        
        return patterns
    
    def find_misconception_chains(
        self,
        df: pd.DataFrame,
        concept_col: str = 'concepts'
    ) -> List[Tuple[str, str, float]]:
        """
        Find misconception chains (one misconception leading to another).
        
        Args:
            df: DataFrame with questions and concepts
            concept_col: Column name containing concepts
            
        Returns:
            List of (concept1, concept2, correlation) tuples
        """
        df = df.copy()
        df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
        
        # Get low competency questions
        low_comp_df = df[df['competency_score'] < 0.3]
        
        if concept_col not in low_comp_df.columns:
            return []
        
        # Extract concepts and find co-occurrences
        concept_pairs = defaultdict(int)
        
        for _, row in low_comp_df.iterrows():
            concepts_str = str(row.get(concept_col, ''))
            if concepts_str:
                concepts = [c.strip() for c in concepts_str.split(',')]
                # Create pairs
                for i, c1 in enumerate(concepts):
                    for c2 in concepts[i+1:]:
                        pair = tuple(sorted([c1, c2]))
                        concept_pairs[pair] += 1
        
        # Return pairs with high co-occurrence
        chains = [
            (c1, c2, count) 
            for (c1, c2), count in concept_pairs.items() 
            if count >= 3
        ]
        
        return sorted(chains, key=lambda x: x[2], reverse=True)
    
    def cluster_error_patterns(
        self,
        questions: List[str],
        embeddings: np.ndarray = None,
        eps: float = 0.5,
        min_samples: int = 3
    ) -> np.ndarray:
        """
        Cluster questions by error patterns using DBSCAN.
        
        Args:
            questions: List of question texts
            embeddings: Pre-computed embeddings (optional)
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            Cluster labels
        """
        if embeddings is None:
            embeddings = self.question_analyzer.embedder.embed_questions(questions)
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Cluster
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        clusters = clustering.fit_predict(embeddings_scaled)
        
        return clusters
