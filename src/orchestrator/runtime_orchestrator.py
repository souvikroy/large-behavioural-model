"""Runtime orchestrator for coordinating API workflows."""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from .service_manager import ServiceManager

logger = logging.getLogger(__name__)


class RuntimeOrchestrator:
    """Coordinate runtime workflows for API requests."""
    
    def __init__(self, service_manager: ServiceManager):
        """
        Initialize runtime orchestrator.
        
        Args:
            service_manager: ServiceManager instance
        """
        self.service_manager = service_manager
    
    def recommend_questions(
        self,
        student_profile: Dict[str, Any],
        target_objectives: Optional[List[str]] = None,
        num_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Coordinate question recommendation workflow.
        
        Args:
            student_profile: Student competency profile
            target_objectives: Optional list of target learning objectives
            num_recommendations: Number of recommendations
            
        Returns:
            List of recommended questions
        """
        logger.info(f"Recommending questions for student profile")
        
        try:
            question_recommender = self.service_manager.get_service('question_recommender')
            
            # Get recommendations
            recommendations = question_recommender.recommend(
                student_profile=student_profile,
                target_objectives=target_objectives,
                num_recommendations=num_recommendations
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Question recommendation failed: {e}")
            raise
    
    def analyze_student(
        self,
        student_responses: List[Dict[str, Any]],
        competency_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Coordinate student analysis workflow (gaps + misconceptions).
        
        Args:
            student_responses: List of student response dictionaries
            competency_threshold: Threshold for identifying gaps
            
        Returns:
            Comprehensive student analysis
        """
        logger.info("Analyzing student performance...")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(student_responses)
            
            # Analyze gaps
            gap_analyzer = self.service_manager.get_service('gap_analyzer')
            gaps = gap_analyzer.analyze_gaps(df, competency_threshold)
            remediation = gap_analyzer.generate_remediation_strategy(gaps)
            
            # Detect misconceptions
            misconception_analyzer = self.service_manager.get_service('misconception_analyzer')
            misconception_detector = self.service_manager.get_service('misconception_detector')
            
            # Detect misconceptions from data
            df_with_misconceptions = misconception_detector.detect_misconceptions_from_data(df)
            
            # Analyze misconceptions
            misconception_analysis = misconception_analyzer.analyze_student_misconceptions(df_with_misconceptions)
            misconception_remediation = misconception_analyzer.recommend_remediation(
                [mc['misconception_id'] for mc in misconception_analysis['misconceptions']]
            )
            
            # Combine results
            analysis = {
                'gaps': gaps,
                'gap_remediation': remediation,
                'misconceptions': misconception_analysis,
                'misconception_remediation': misconception_remediation,
                'summary': {
                    'total_gaps': len(gaps.get('by_chapter', {})),
                    'total_misconceptions': misconception_analysis['total_identified'],
                    'priority_areas': remediation.get('priority_areas', [])[:3]
                }
            }
            
            logger.info("Student analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Student analysis failed: {e}")
            raise
    
    def generate_learning_path(
        self,
        student_profile: Dict[str, Any],
        target_concept: Optional[str] = None,
        target_chapter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Coordinate learning path generation.
        
        Args:
            student_profile: Student competency profile
            target_concept: Optional target concept
            target_chapter: Optional target chapter
            
        Returns:
            Learning path
        """
        logger.info("Generating learning path...")
        
        try:
            learning_path_generator = self.service_manager.get_service('learning_path_generator')
            
            path = learning_path_generator.generate_path(
                student_profile=student_profile,
                target_concept=target_concept,
                target_chapter=target_chapter
            )
            
            logger.info(f"Generated learning path with {len(path)} steps")
            return path
            
        except Exception as e:
            logger.error(f"Learning path generation failed: {e}")
            raise
    
    def predict_competency(
        self,
        question_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate competency prediction.
        
        Args:
            question_data: Question data dictionary
            
        Returns:
            Prediction result with competency score
        """
        logger.info("Predicting competency...")
        
        try:
            competency_predictor = self.service_manager.get_service('competency_predictor')
            
            # Convert to DataFrame
            question_df = pd.DataFrame([question_data])
            
            # Predict
            predictions = competency_predictor.predict(question_df)
            
            result = {
                'predicted_competency': float(predictions[0]) if len(predictions) > 0 else 0.0,
                'confidence': None  # Could be added if model provides it
            }
            
            logger.info(f"Predicted competency: {result['predicted_competency']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Competency prediction failed: {e}")
            raise
    
    def get_concept_info(self, concept_id: str) -> Dict[str, Any]:
        """
        Get concept information from knowledge graph.
        
        Args:
            concept_id: Concept identifier
            
        Returns:
            Concept details
        """
        logger.info(f"Getting concept info for {concept_id}...")
        
        try:
            knowledge_graph_query = self.service_manager.get_service('knowledge_graph_query')
            
            details = knowledge_graph_query.get_concept_details(concept_id)
            
            return {'concept_details': details}
            
        except Exception as e:
            logger.error(f"Failed to get concept info: {e}")
            raise
    
    def get_prerequisites(self, concept_id: str, max_depth: int = 5) -> List[str]:
        """
        Get prerequisites for a concept.
        
        Args:
            concept_id: Concept identifier
            max_depth: Maximum depth to traverse
            
        Returns:
            List of prerequisites
        """
        logger.info(f"Getting prerequisites for {concept_id}...")
        
        try:
            knowledge_graph_query = self.service_manager.get_service('knowledge_graph_query')
            
            prerequisites = knowledge_graph_query.find_prerequisites(concept_id, max_depth)
            
            return prerequisites
            
        except Exception as e:
            logger.error(f"Failed to get prerequisites: {e}")
            raise
    
    def find_learning_path_between_concepts(
        self,
        from_concept: str,
        to_concept: str,
        respect_prerequisites: bool = True
    ) -> List[str]:
        """
        Find learning path between two concepts.
        
        Args:
            from_concept: Starting concept
            to_concept: Target concept
            respect_prerequisites: Whether to respect prerequisites
            
        Returns:
            Learning path as list of concept IDs
        """
        logger.info(f"Finding path from {from_concept} to {to_concept}...")
        
        try:
            knowledge_graph_query = self.service_manager.get_service('knowledge_graph_query')
            
            path = knowledge_graph_query.find_learning_path(
                from_concept,
                to_concept,
                respect_prerequisites
            )
            
            return path
            
        except Exception as e:
            logger.error(f"Failed to find learning path: {e}")
            raise
    
    def recommend_for_misconceptions(
        self,
        misconceptions: List[str],
        num_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend questions to address misconceptions.
        
        Args:
            misconceptions: List of misconception IDs
            num_recommendations: Number of recommendations
            
        Returns:
            List of recommended questions
        """
        logger.info(f"Recommending questions for {len(misconceptions)} misconceptions...")
        
        try:
            question_recommender = self.service_manager.get_service('question_recommender')
            
            recommendations = question_recommender.recommend_for_misconceptions(
                misconceptions,
                num_recommendations
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Misconception recommendation failed: {e}")
            raise
