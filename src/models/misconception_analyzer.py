"""Misconception analyzer for students."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yaml
import json
import logging

from ..knowledge_graph.query import KnowledgeGraphQuery
from ..models.misconception_detector import MisconceptionDetector

logger = logging.getLogger(__name__)

# Try to import OpenAI components
try:
    from ..openai.client import OpenAIClient
    from ..openai.cache import OpenAICache
    from ..openai.prompts import get_misconception_remediation_prompt
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIClient = None
    OpenAICache = None
    get_misconception_remediation_prompt = None


class MisconceptionAnalyzer:
    """Analyze and predict misconceptions for students."""
    
    def __init__(
        self,
        misconception_detector: MisconceptionDetector,
        knowledge_graph_query: KnowledgeGraphQuery,
        config_path: str = "config.yaml"
    ):
        """
        Initialize misconception analyzer.
        
        Args:
            misconception_detector: MisconceptionDetector instance
            knowledge_graph_query: KnowledgeGraphQuery instance
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.misconception_detector = misconception_detector
        self.graph_query = knowledge_graph_query
        
        # Initialize OpenAI support if available
        self._openai_enabled = False
        self._openai_client = None
        self._openai_cache = None
        
        if OPENAI_AVAILABLE:
            try:
                openai_config = self.config.get('openai', {})
                if openai_config.get('use_openai_misconceptions', False):
                    self._openai_client = OpenAIClient(
                        api_url=openai_config.get('api_url', 'http://192.168.0.4:1601'),
                        model=openai_config.get('model', 'qwen/qwen3-4b-thinking-2507'),
                        timeout=openai_config.get('timeout', 30),
                        max_retries=openai_config.get('max_retries', 3)
                    )
                    
                    if self._openai_client.is_available():
                        self._openai_enabled = True
                        self._openai_cache = OpenAICache(
                            max_size=openai_config.get('cache_max_size', 1000),
                            ttl=openai_config.get('cache_ttl', 3600)
                        )
                        logger.info("OpenAI misconception remediation enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI misconception remediation: {e}")
    
    def analyze_student_misconceptions(
        self,
        student_responses: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Analyze misconceptions from student responses.
        
        Args:
            student_responses: DataFrame with student question responses
            
        Returns:
            Dictionary with identified misconceptions
        """
        student_responses = student_responses.copy()
        student_responses['competency_score'] = pd.to_numeric(
            student_responses['competency_score'], errors='coerce'
        )
        
        # Identify low-performing questions
        low_comp_threshold = 0.3
        low_comp_questions = student_responses[
            student_responses['competency_score'] < low_comp_threshold
        ]
        
        identified_misconceptions = []
        
        # Match against known misconceptions
        for idx, row in low_comp_questions.iterrows():
            question_id = row.get('question_id')
            
            # Get misconceptions for this question
            misconceptions = self.misconception_detector.get_misconceptions_for_question(idx)
            
            for mc_id in misconceptions:
                mc_data = self.misconception_detector.misconceptions.get(mc_id, {})
                
                identified_misconceptions.append({
                    'misconception_id': mc_id,
                    'concept': mc_data.get('concept', ''),
                    'type': mc_data.get('type', 'Conceptual'),
                    'confidence': 1.0 - row['competency_score'],  # Higher confidence for lower competency
                    'question_id': question_id,
                    'severity': mc_data.get('severity', 0.0)
                })
        
        # Remove duplicates and aggregate
        unique_misconceptions = {}
        for mc in identified_misconceptions:
            mc_id = mc['misconception_id']
            if mc_id not in unique_misconceptions:
                unique_misconceptions[mc_id] = mc
                unique_misconceptions[mc_id]['frequency'] = 1
            else:
                unique_misconceptions[mc_id]['frequency'] += 1
                # Update confidence (average)
                unique_misconceptions[mc_id]['confidence'] = (
                    unique_misconceptions[mc_id]['confidence'] + mc['confidence']
                ) / 2
        
        return {
            'misconceptions': list(unique_misconceptions.values()),
            'total_identified': len(unique_misconceptions)
        }
    
    def find_misconception_chains(
        self,
        misconceptions: List[str]
    ) -> List[Dict[str, any]]:
        """
        Find misconception chains (one leading to another).
        
        Args:
            misconceptions: List of misconception IDs
            
        Returns:
            List of misconception chains
        """
        chains = []
        
        if self.graph_query is None:
            return chains
        
        # Find relationships between misconceptions
        for mc_id in misconceptions:
            mc_node = f"misconception_{mc_id}"
            
            if mc_node in self.graph_query.graph:
                # Find related misconceptions
                for u, v, data in self.graph_query.graph.edges(mc_node, data=True):
                    if data.get('edge_type') == 'LEADS_TO':
                        target_mc = v
                        chains.append({
                            'source_misconception': mc_id,
                            'target_misconception': target_mc.replace('misconception_', ''),
                            'relationship': 'leads_to'
                        })
        
        return chains
    
    def recommend_remediation(
        self,
        misconceptions: List[str]
    ) -> Dict[str, any]:
        """
        Recommend remediation for misconceptions.
        
        Args:
            misconceptions: List of misconception IDs
            
        Returns:
            Dictionary with remediation recommendations
        """
        # Try OpenAI first if enabled
        if self._openai_enabled and get_misconception_remediation_prompt:
            try:
                # Collect misconception data
                misconception_data = []
                for mc_id in misconceptions:
                    mc_data = self.misconception_detector.misconceptions.get(mc_id, {})
                    misconception_data.append({
                        'misconception_id': mc_id,
                        'concept': mc_data.get('concept', ''),
                        'type': mc_data.get('type', 'Conceptual'),
                        'severity': mc_data.get('severity', 0.0)
                    })
                
                if misconception_data:
                    prompt = get_misconception_remediation_prompt(misconception_data)
                    response = self._openai_client.generate(
                        prompt,
                        max_tokens=512,
                        temperature=0.7
                    )
                    
                    if response:
                        try:
                            json_start = response.find('{')
                            json_end = response.rfind('}') + 1
                            if json_start >= 0 and json_end > json_start:
                                json_str = response[json_start:json_end]
                                openai_remediation = json.loads(json_str)
                                
                                # Ensure same structure
                                remediation = {
                                    'questions_to_review': openai_remediation.get('questions_to_review', []),
                                    'prerequisites_to_review': openai_remediation.get('prerequisites_to_review', []),
                                    'concepts_to_clarify': openai_remediation.get('concepts_to_clarify', [])
                                }
                                
                                if remediation['questions_to_review'] or remediation['concepts_to_clarify']:
                                    return remediation
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse OpenAI remediation, using fallback")
            except Exception as e:
                logger.warning(f"OpenAI remediation recommendation failed: {e}, using fallback")
        
        # Fallback to existing implementation
        remediation = {
            'questions_to_review': [],
            'prerequisites_to_review': [],
            'concepts_to_clarify': []
        }
        
        for mc_id in misconceptions:
            mc_data = self.misconception_detector.misconceptions.get(mc_id, {})
            
            # Get associated questions
            question_indices = mc_data.get('associated_questions', [])
            remediation['questions_to_review'].extend(question_indices[:5])  # Limit to 5
            
            # Find prerequisite concepts
            concept = mc_data.get('concept', '')
            if concept and self.graph_query:
                concept_node = f"concept_{hash(concept)}"
                prerequisites = self.graph_query.find_prerequisites(concept_node)
                remediation['prerequisites_to_review'].extend(prerequisites[:3])  # Limit to 3
            
            # Add concept to clarify
            remediation['concepts_to_clarify'].append({
                'concept': concept,
                'misconception_type': mc_data.get('type', 'Conceptual'),
                'reason': f'Addresses misconception: {mc_id}'
            })
        
        return remediation
    
    def predict_misconception_likelihood(
        self,
        student_profile: Dict[str, any],
        concept: str
    ) -> float:
        """
        Predict likelihood of misconception for a concept.
        
        Args:
            student_profile: Student competency profile
            concept: Concept name
            
        Returns:
            Likelihood score (0-1)
        """
        # Get misconceptions for this concept
        misconceptions = self.misconception_detector.get_misconceptions_for_concept(concept)
        
        if not misconceptions:
            return 0.0
        
        # Check student's competency in related areas
        competency_by_chapter = student_profile.get('competency_by_chapter', {})
        
        # Find chapters related to this concept
        related_chapters = [
            ch for ch, comp in competency_by_chapter.items()
            if comp < 0.5  # Low competency
        ]
        
        # Higher likelihood if student has low competency in related areas
        if related_chapters:
            avg_low_comp = np.mean([competency_by_chapter[ch] for ch in related_chapters])
            likelihood = 1.0 - avg_low_comp
        else:
            likelihood = 0.3  # Default moderate likelihood
        
        return min(likelihood, 1.0)
