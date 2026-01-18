"""Adaptive question recommendation system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
from sklearn.metrics.pairwise import cosine_similarity

from .competency_predictor import CompetencyPredictor
from .difficulty_calibrator import DifficultyCalibrator
from ..knowledge_graph.query import KnowledgeGraphQuery


class QuestionRecommender:
    """Recommend questions based on competency and learning objectives."""
    
    def __init__(
        self,
        competency_predictor: CompetencyPredictor,
        difficulty_calibrator: DifficultyCalibrator,
        knowledge_graph_query: Optional[KnowledgeGraphQuery] = None,
        config_path: str = "config.yaml"
    ):
        """
        Initialize question recommender.
        
        Args:
            competency_predictor: CompetencyPredictor instance
            difficulty_calibrator: DifficultyCalibrator instance
            knowledge_graph_query: Optional KnowledgeGraphQuery instance
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.recommender_config = self.config.get('models', {}).get('question_recommender', {})
        self.top_k = self.recommender_config.get('top_k', 10)
        self.use_graph_filtering = self.recommender_config.get('use_graph_filtering', True)
        self.use_content_filtering = self.recommender_config.get('use_content_filtering', True)
        
        self.competency_predictor = competency_predictor
        self.difficulty_calibrator = difficulty_calibrator
        self.graph_query = knowledge_graph_query
        
        self.question_pool = None
        self.question_embeddings = None
    
    def set_question_pool(self, df: pd.DataFrame) -> None:
        """
        Set the pool of questions to recommend from.
        
        Args:
            df: DataFrame with questions
        """
        self.question_pool = df.copy()
        
        # Pre-compute embeddings if using content filtering
        if self.use_content_filtering and 'question_text' in df.columns:
            from ..nlp.text_embeddings import TextEmbedder
            embedder = TextEmbedder()
            question_texts = df['question_text'].astype(str).tolist()
            self.question_embeddings = embedder.embed_questions(question_texts)
    
    def recommend(
        self,
        student_profile: Dict[str, any],
        target_objectives: Optional[List[str]] = None,
        num_recommendations: int = None
    ) -> List[Dict[str, any]]:
        """
        Recommend questions for a student.
        
        Args:
            student_profile: Dictionary with student competency profile
            target_objectives: Optional list of target learning objectives
            num_recommendations: Number of recommendations (uses top_k if None)
            
        Returns:
            List of recommended questions with scores and reasoning
        """
        if self.question_pool is None:
            raise ValueError("Question pool not set. Call set_question_pool() first.")
        
        num_rec = num_recommendations or self.top_k
        
        # Filter questions
        filtered_questions = self._filter_questions(student_profile, target_objectives)
        
        if len(filtered_questions) == 0:
            return []
        
        # Score questions
        scored_questions = self._score_questions(filtered_questions, student_profile)
        
        # Sort by score and return top K
        scored_questions.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_questions[:num_rec]
    
    def _filter_questions(
        self,
        student_profile: Dict[str, any],
        target_objectives: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Filter questions based on student profile and objectives.
        
        Args:
            student_profile: Student competency profile
            target_objectives: Target learning objectives
            
        Returns:
            Filtered DataFrame
        """
        filtered = self.question_pool.copy()
        
        # Graph-based filtering
        if self.use_graph_filtering and self.graph_query:
            # Filter by prerequisites
            mastered_concepts = student_profile.get('mastered_concepts', [])
            
            # Only include questions testing concepts with met prerequisites
            # This is simplified - in practice, you'd check each question's concepts
            pass
        
        # Content-based filtering
        if self.use_content_filtering and target_objectives:
            # Filter by target objectives (simplified)
            if 'learning_objective' in filtered.columns:
                filtered = filtered[
                    filtered['learning_objective'].isin(target_objectives)
                ]
        
        # Filter by difficulty range
        current_competency = student_profile.get('avg_competency', 0.5)
        difficulty_range = (
            max(0, current_competency - 0.2),
            min(1, current_competency + 0.2)
        )
        
        # Predict difficulty for all questions
        if '3PL' not in filtered.columns or filtered['3PL'].isna().any():
            predicted_difficulties = self.difficulty_calibrator.predict(filtered)
            filtered['predicted_3pl'] = predicted_difficulties
        
        # Filter by competency level
        weak_areas = student_profile.get('weak_areas', {})
        if weak_areas:
            # Prioritize questions in weak areas
            pass
        
        return filtered
    
    def _score_questions(
        self,
        questions: pd.DataFrame,
        student_profile: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """
        Score questions for recommendation.
        
        Args:
            questions: DataFrame with questions
            student_profile: Student competency profile
            
        Returns:
            List of scored questions
        """
        scored = []
        
        # Batch predict competencies for all questions at once
        predicted_competencies = self.competency_predictor.predict(questions)
        
        # Process all questions (same logic, just batched predictions)
        target_competency = student_profile.get('target_competency', 0.6)
        weak_areas = student_profile.get('weak_areas', {})
        
        for idx, (_, row) in enumerate(questions.iterrows()):
            score = 0.0
            reasoning = []
            
            # Use batch-predicted competency
            predicted_competency = predicted_competencies[idx] if idx < len(predicted_competencies) else 0.5
            
            # Score based on predicted competency (prefer moderate difficulty)
            competency_diff = abs(predicted_competency - target_competency)
            score += (1.0 - competency_diff) * 0.4
            reasoning.append(f"Predicted competency: {predicted_competency:.2f}")
            
            # Score based on difficulty appropriateness
            if '3PL' in row and not pd.isna(row['3PL']):
                difficulty = row['3PL']
                # Prefer questions with moderate difficulty
                if -2.0 <= difficulty <= 2.0:
                    score += 0.3
                    reasoning.append(f"Appropriate difficulty: {difficulty:.2f}")
            
            # Score based on weak areas
            if weak_areas:
                subject = row.get('Subject', '')
                chapter = row.get('chapter', '')
                
                if subject in weak_areas:
                    score += 0.2
                    reasoning.append(f"Addresses weak area: {subject}")
                if chapter in weak_areas.get(subject, []):
                    score += 0.1
                    reasoning.append(f"Addresses weak chapter: {chapter}")
            
            scored.append({
                'question_id': row.get('question_id'),
                'question_text': row.get('question_text', '')[:100],
                'score': score,
                'predicted_competency': predicted_competency,
                'reasoning': '; '.join(reasoning),
                'subject': row.get('Subject'),
                'chapter': row.get('chapter'),
                'bloom_tag': row.get('Bloom_tag')
            })
        
        return scored
    
    def recommend_for_misconceptions(
        self,
        misconceptions: List[str],
        num_recommendations: int = None
    ) -> List[Dict[str, any]]:
        """
        Recommend questions to address specific misconceptions.
        
        Args:
            misconceptions: List of misconception IDs
            num_recommendations: Number of recommendations
            
        Returns:
            List of recommended questions
        """
        if self.question_pool is None:
            raise ValueError("Question pool not set.")
        
        num_rec = num_recommendations or self.top_k
        
        # Filter questions that address misconceptions
        # This is simplified - in practice, you'd use misconception-question mappings
        filtered = self.question_pool.copy()
        
        # Score and return
        student_profile = {'target_competency': 0.7}  # Higher for remediation
        scored = self._score_questions(filtered, student_profile)
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        return scored[:num_rec]
