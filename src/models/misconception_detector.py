"""Misconception detection system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import yaml

from ..nlp.question_analyzer import QuestionAnalyzer
from ..nlp.concept_extractor import ConceptExtractor


class MisconceptionDetector:
    """Detect and catalog misconceptions from question-response data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize misconception detector.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.nlp_config = self.config.get('nlp', {})
        self.question_analyzer = QuestionAnalyzer(self.nlp_config, config_path=config_path)
        self.concept_extractor = ConceptExtractor(config_path=config_path)
        
        # Misconception database
        self.misconceptions = {}
        self.misconception_question_map = defaultdict(list)
        self.concept_misconception_map = defaultdict(list)
    
    def detect_misconceptions_from_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect misconceptions from question-response data.
        
        Args:
            df: DataFrame with questions and competency scores
            
        Returns:
            DataFrame with misconception indicators
        """
        df = df.copy()
        
        # Ensure competency_score is numeric
        df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
        
        # Analyze low competency questions
        competency_threshold = self.config.get('preprocessing', {}).get('min_competency_threshold', 0.3)
        df = self.question_analyzer.analyze_low_competency_questions(df, competency_threshold)
        
        # Identify misconception patterns
        patterns = self.question_analyzer.identify_misconception_patterns(df, competency_threshold)
        
        # Mark questions with high misconception risk
        df['misconception_risk'] = 0.0
        
        # Calculate risk based on indicators
        indicator_cols = [col for col in df.columns if col.startswith('contains_') or col.startswith('asks_')]
        if indicator_cols:
            df['misconception_risk'] = df[indicator_cols].sum(axis=1) / len(indicator_cols)
        
        # Increase risk for low competency questions
        low_comp_mask = df['competency_score'] < competency_threshold
        df.loc[low_comp_mask, 'misconception_risk'] += 0.5
        df['misconception_risk'] = df['misconception_risk'].clip(0, 1)
        
        return df
    
    def catalog_misconceptions(
        self, 
        df: pd.DataFrame,
        min_frequency: int = 3
    ) -> Dict[str, Dict]:
        """
        Catalog misconceptions from data.
        
        Args:
            df: DataFrame with misconception analysis
            min_frequency: Minimum frequency to consider a misconception
            
        Returns:
            Dictionary of cataloged misconceptions
        """
        df = df.copy()
        df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
        
        competency_threshold = self.config.get('preprocessing', {}).get('min_competency_threshold', 0.3)
        low_comp_df = df[df['competency_score'] < competency_threshold]
        
        misconceptions = {}
        
        # Group by concept patterns
        concept_groups = defaultdict(list)
        for idx, row in low_comp_df.iterrows():
            concepts = self.concept_extractor.extract_concepts(row.get('question_text', ''))
            for concept in concepts:
                concept_groups[concept].append(idx)
        
        # Identify misconceptions
        misconception_id = 0
        for concept, question_indices in concept_groups.items():
            if len(question_indices) >= min_frequency:
                misconception_id += 1
                misconception_key = f"MC_{misconception_id}"
                
                # Get question details
                questions = low_comp_df.loc[question_indices]
                
                # Determine misconception type
                misconception_type = self._classify_misconception_type(questions)
                
                misconceptions[misconception_key] = {
                    'concept': concept,
                    'type': misconception_type,
                    'frequency': len(question_indices),
                    'severity': questions['competency_score'].mean(),
                    'associated_questions': question_indices.tolist(),
                    'subjects': questions['Subject'].value_counts().to_dict() if 'Subject' in questions.columns else {},
                    'chapters': questions['chapter'].value_counts().to_dict() if 'chapter' in questions.columns else {},
                    'bloom_levels': questions['Bloom_tag'].value_counts().to_dict() if 'Bloom_tag' in questions.columns else {}
                }
                
                # Update mappings
                for q_idx in question_indices:
                    self.misconception_question_map[q_idx].append(misconception_key)
                self.concept_misconception_map[concept].append(misconception_key)
        
        self.misconceptions = misconceptions
        return misconceptions
    
    def _classify_misconception_type(self, questions: pd.DataFrame) -> str:
        """
        Classify misconception type based on question characteristics.
        
        Args:
            questions: DataFrame of questions with same misconception
            
        Returns:
            Misconception type
        """
        # Check Bloom taxonomy levels
        if 'Bloom_tag' in questions.columns:
            bloom_dist = questions['Bloom_tag'].value_counts()
            if 'Apply' in bloom_dist.index or 'Analyse' in bloom_dist.index:
                return 'Procedural'
            elif 'Remember' in bloom_dist.index or 'Understand' in bloom_dist.index:
                return 'Conceptual'
        
        # Default to conceptual
        return 'Conceptual'
    
    def get_misconceptions_for_concept(self, concept: str) -> List[str]:
        """
        Get misconceptions associated with a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            List of misconception IDs
        """
        return self.concept_misconception_map.get(concept, [])
    
    def get_misconceptions_for_question(self, question_id: int) -> List[str]:
        """
        Get misconceptions associated with a question.
        
        Args:
            question_id: Question index
            
        Returns:
            List of misconception IDs
        """
        return self.misconception_question_map.get(question_id, [])
