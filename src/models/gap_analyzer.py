"""Competency gap analyzer."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yaml
import json
import logging

from .competency_predictor import CompetencyPredictor
from ..knowledge_graph.query import KnowledgeGraphQuery

logger = logging.getLogger(__name__)

# Try to import LLM components
try:
    from ..llm.client import LLMClient
    from ..llm.cache import LLMCache
    from ..llm.prompts import get_gap_analysis_prompt
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMClient = None
    LLMCache = None
    get_gap_analysis_prompt = None


class GapAnalyzer:
    """Analyze competency gaps and identify root causes."""
    
    def __init__(
        self,
        competency_predictor: CompetencyPredictor,
        knowledge_graph_query: KnowledgeGraphQuery,
        config_path: str = "config.yaml"
    ):
        """
        Initialize gap analyzer.
        
        Args:
            competency_predictor: CompetencyPredictor instance
            knowledge_graph_query: KnowledgeGraphQuery instance
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.competency_predictor = competency_predictor
        self.graph_query = knowledge_graph_query
        
        # Initialize LLM support if available
        self._llm_enabled = False
        self._llm_client = None
        self._llm_cache = None
        
        if LLM_AVAILABLE:
            try:
                llm_config = self.config.get('llm', {})
                if llm_config.get('use_llm_misconceptions', False):  # Reuse same flag
                    self._llm_client = LLMClient(
                        api_url=llm_config.get('api_url', 'http://192.168.0.4:1601'),
                        model=llm_config.get('model', 'qwen/qwen3-4b-thinking-2507'),
                        timeout=llm_config.get('timeout', 30),
                        max_retries=llm_config.get('max_retries', 3)
                    )
                    
                    if self._llm_client.is_available():
                        self._llm_enabled = True
                        self._llm_cache = LLMCache(
                            max_size=llm_config.get('cache_max_size', 1000),
                            ttl=llm_config.get('cache_ttl', 3600)
                        )
                        logger.info("LLM gap analysis enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM gap analysis: {e}")
    
    def analyze_gaps(
        self,
        df: pd.DataFrame,
        competency_threshold: float = 0.6
    ) -> Dict[str, any]:
        """
        Analyze competency gaps by dimension.
        
        Args:
            df: DataFrame with questions and competency scores
            competency_threshold: Threshold for identifying gaps
            
        Returns:
            Dictionary with gap analysis results
        """
        df = df.copy()
        df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
        
        gaps = {}
        
        # Analyze by subject
        if 'Subject' in df.columns:
            gaps['by_subject'] = self._analyze_by_dimension(df, 'Subject', competency_threshold)
        
        # Analyze by chapter
        if 'chapter' in df.columns:
            gaps['by_chapter'] = self._analyze_by_dimension(df, 'chapter', competency_threshold)
        
        # Analyze by learning objective
        if 'learning_objective' in df.columns:
            gaps['by_learning_objective'] = self._analyze_by_dimension(
                df, 'learning_objective', competency_threshold
            )
        
        # Analyze by Bloom level
        if 'Bloom_tag' in df.columns:
            gaps['by_bloom'] = self._analyze_by_dimension(df, 'Bloom_tag', competency_threshold)
        
        # Identify root causes
        gaps['root_causes'] = self._identify_root_causes(df, gaps)
        
        # Identify misconceptions
        gaps['misconceptions'] = self._identify_misconceptions(df)
        
        return gaps
    
    def _analyze_by_dimension(
        self,
        df: pd.DataFrame,
        dimension: str,
        threshold: float
    ) -> Dict[str, Dict]:
        """
        Analyze gaps by a specific dimension.
        
        Args:
            df: DataFrame with data
            dimension: Dimension to analyze
            threshold: Competency threshold
            
        Returns:
            Dictionary with gap analysis by dimension value
        """
        analysis = {}
        
        for value in df[dimension].unique():
            value_df = df[df[dimension] == value]
            avg_competency = value_df['competency_score'].mean()
            
            if avg_competency < threshold:
                analysis[value] = {
                    'avg_competency': avg_competency,
                    'total_questions': len(value_df),
                    'low_competency_count': len(value_df[value_df['competency_score'] < threshold]),
                    'gap_severity': threshold - avg_competency,
                    'rank': threshold - avg_competency  # For sorting
                }
        
        # Sort by severity
        sorted_analysis = dict(
            sorted(analysis.items(), key=lambda x: x[1]['rank'], reverse=True)
        )
        
        return sorted_analysis
    
    def _identify_root_causes(
        self,
        df: pd.DataFrame,
        gaps: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """
        Identify root causes of gaps using knowledge graph.
        
        Args:
            df: DataFrame with data
            gaps: Gap analysis results
            
        Returns:
            List of root causes
        """
        root_causes = []
        
        if self.graph_query is None:
            return root_causes
        
        # For each weak area, find prerequisites
        weak_chapters = gaps.get('by_chapter', {})
        
        for chapter, chapter_data in list(weak_chapters.items())[:10]:  # Limit to top 10
            chapter_node = f"chapter_{chapter}"
            
            if chapter_node in self.graph_query.graph:
                # Find prerequisites
                prerequisites = self.graph_query.find_prerequisites(chapter_node)
                
                # Check if prerequisites are also weak
                for prereq in prerequisites:
                    prereq_data = self.graph_query.graph.nodes[prereq]
                    prereq_name = prereq_data.get('name', prereq)
                    
                    # Check if this prerequisite appears in weak areas
                    is_weak = False
                    for weak_ch, weak_data in weak_chapters.items():
                        if prereq_name in str(weak_ch):
                            is_weak = True
                            break
                    
                    if is_weak:
                        root_causes.append({
                            'type': 'prerequisite_gap',
                            'chapter': chapter,
                            'prerequisite': prereq_name,
                            'severity': chapter_data['gap_severity']
                        })
        
        return root_causes
    
    def _identify_misconceptions(self, df: pd.DataFrame) -> List[Dict[str, any]]:
        """
        Identify misconceptions causing gaps.
        
        Args:
            df: DataFrame with data
            
        Returns:
            List of misconceptions
        """
        misconceptions = []
        
        # Find questions with low competency
        low_comp_df = df[df['competency_score'] < 0.3]
        
        if len(low_comp_df) == 0:
            return misconceptions
        
        # Group by concept/chapter to find patterns
        if 'chapter' in low_comp_df.columns:
            chapter_counts = low_comp_df['chapter'].value_counts()
            
            for chapter, count in chapter_counts.items():
                if count >= 3:  # Threshold for misconception
                    misconceptions.append({
                        'type': 'conceptual_misconception',
                        'chapter': chapter,
                        'frequency': count,
                        'avg_competency': low_comp_df[low_comp_df['chapter'] == chapter]['competency_score'].mean()
                    })
        
        return misconceptions
    
    def generate_remediation_strategy(
        self,
        gaps: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Generate remediation strategy based on gap analysis.
        
        Args:
            gaps: Gap analysis results
            
        Returns:
            Dictionary with remediation strategy
        """
        # Try LLM first if enabled
        if self._llm_enabled and get_gap_analysis_prompt:
            try:
                prompt = get_gap_analysis_prompt(gaps)
                response = self._llm_client.generate(
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
                            llm_strategy = json.loads(json_str)
                            
                            # Ensure same structure as existing method
                            strategy = {
                                'priority_areas': llm_strategy.get('priority_areas', []),
                                'recommended_actions': llm_strategy.get('recommended_actions', []),
                                'prerequisite_review': llm_strategy.get('prerequisite_review', [])
                            }
                            
                            # Validate and format to match expected structure
                            if strategy['priority_areas'] or strategy['recommended_actions']:
                                return strategy
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse LLM remediation strategy, using fallback")
            except Exception as e:
                logger.warning(f"LLM remediation strategy generation failed: {e}, using fallback")
        
        # Fallback to existing implementation
        strategy = {
            'priority_areas': [],
            'recommended_actions': [],
            'prerequisite_review': []
        }
        
        # Identify priority areas (most severe gaps)
        by_chapter = gaps.get('by_chapter', {})
        priority_chapters = list(by_chapter.items())[:5]  # Top 5
        
        for chapter, data in priority_chapters:
            strategy['priority_areas'].append({
                'chapter': chapter,
                'severity': data['gap_severity'],
                'avg_competency': data['avg_competency']
            })
            
            strategy['recommended_actions'].append({
                'action': f'Focus on {chapter}',
                'reason': f'Low competency ({data["avg_competency"]:.2f})',
                'priority': 'high' if data['gap_severity'] > 0.3 else 'medium'
            })
        
        # Add prerequisite review
        root_causes = gaps.get('root_causes', [])
        for root_cause in root_causes[:5]:
            strategy['prerequisite_review'].append({
                'concept': root_cause.get('prerequisite'),
                'reason': f'Prerequisite gap affecting {root_cause.get("chapter")}'
            })
        
        return strategy
