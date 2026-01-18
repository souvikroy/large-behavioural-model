"""Question text analysis for misconception detection."""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Optional
from collections import Counter
from sklearn.cluster import KMeans
import json
import yaml
import logging

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    BERTopic = None

from .text_embeddings import TextEmbedder
from .concept_extractor import ConceptExtractor

logger = logging.getLogger(__name__)

# Try to import LLM components
try:
    from ..llm.client import LLMClient
    from ..llm.cache import LLMCache
    from ..llm.prompts import get_misconception_detection_prompt
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMClient = None
    LLMCache = None
    get_misconception_detection_prompt = None


class QuestionAnalyzer:
    """Analyze question text to identify misconception patterns."""
    
    def __init__(self, config: Dict = None, config_path: str = "config.yaml"):
        """
        Initialize question analyzer.
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration file
        """
        self.config = config or {}
        self.config_path = config_path
        self.embedder = TextEmbedder(
            model_name=self.config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
            config_path=config_path
        )
        self.concept_extractor = ConceptExtractor(config_path=config_path)
        self.misconception_keywords = self.config.get(
            'misconception_keywords',
            ['cannot', 'incorrect', 'wrong', 'mistake', 'error', 'identify the mistake', 'find the error']
        )
        
        # Initialize LLM support if available
        self._llm_enabled = False
        self._llm_client = None
        self._llm_cache = None
        
        if LLM_AVAILABLE:
            try:
                with open(config_path, 'r') as f:
                    llm_config = yaml.safe_load(f).get('llm', {})
                
                if llm_config.get('use_llm_misconceptions', False):
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
                        logger.info("LLM misconception detection enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM misconception detection: {e}")
    
    def _llm_detect_indicators(self, question_text: str) -> Optional[Dict[str, bool]]:
        """
        Detect misconception indicators using LLM API (internal method).
        
        Args:
            question_text: Question text to analyze
            
        Returns:
            Dictionary of indicator flags or None if failed
        """
        if not self._llm_enabled or not self._llm_client or not get_misconception_detection_prompt:
            return None
        
        # Check cache first
        if self._llm_cache:
            cache_key = f"misconception:{question_text}"
            cached = self._llm_cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            prompt = get_misconception_detection_prompt(question_text)
            response = self._llm_client.generate(
                prompt,
                max_tokens=256,
                temperature=0.3
            )
            
            if response:
                # Try to parse JSON response
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        analysis = json.loads(json_str)
                        
                        # Convert to same format as keyword-based detection
                        indicators = {}
                        has_indicators = analysis.get('has_misconception_indicators', False)
                        misconception_type = analysis.get('misconception_type', 'none')
                        
                        # Map to existing indicator format
                        for keyword in self.misconception_keywords:
                            indicators[f'contains_{keyword}'] = keyword in question_text.lower()
                        
                        indicators['asks_for_mistake'] = has_indicators and misconception_type != 'none'
                        indicators['compares_approaches'] = 'compare' in question_text.lower() or 'difference' in question_text.lower()
                        
                        # Cache result
                        if self._llm_cache:
                            cache_key = f"misconception:{question_text}"
                            self._llm_cache.set(cache_key, indicators)
                        
                        return indicators
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.warning(f"LLM misconception detection failed: {e}")
        
        return None
    
    def detect_misconception_indicators(self, question_text: str) -> Dict[str, bool]:
        """
        Detect misconception indicators in question text.
        
        Args:
            question_text: Question text to analyze
            
        Returns:
            Dictionary of indicator flags
        """
        # Try LLM first if enabled
        if self._llm_enabled:
            try:
                llm_indicators = self._llm_detect_indicators(question_text)
                if llm_indicators is not None:
                    return llm_indicators
            except Exception as e:
                logger.warning(f"LLM misconception detection failed, falling back to keyword matching: {e}")
        
        # Fallback to keyword-based detection (existing implementation)
        text_lower = str(question_text).lower()
        
        indicators = {}
        for keyword in self.misconception_keywords:
            indicators[f'contains_{keyword}'] = keyword in text_lower
        
        # Additional patterns
        indicators['asks_for_mistake'] = any(
            phrase in text_lower 
            for phrase in ['identify the mistake', 'find the error', 'what is wrong', 'what is incorrect']
        )
        
        indicators['compares_approaches'] = any(
            phrase in text_lower 
            for phrase in ['which is correct', 'which statement', 'compare', 'difference between']
        )
        
        return indicators
    
    def analyze_question_intent(self, question_text: str) -> Dict[str, any]:
        """
        Analyze question intent and characteristics.
        
        Args:
            question_text: Question text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Get embedding
        embedding = self.embedder.embed(question_text)
        
        # Extract concepts
        concepts = self.concept_extractor.extract_concepts(question_text)
        
        # Detect misconception indicators
        indicators = self.detect_misconception_indicators(question_text)
        
        return {
            'embedding': embedding,
            'concepts': concepts,
            'indicators': indicators,
            'text_length': len(question_text),
            'word_count': len(str(question_text).split())
        }
    
    def cluster_questions_by_pattern(self, questions: List[str], n_clusters: int = 10) -> np.ndarray:
        """
        Cluster questions by semantic similarity to find error patterns.
        
        Args:
            questions: List of question texts
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels for each question
        """
        # Get embeddings
        embeddings = self.embedder.embed_questions(questions)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        return clusters
    
    def topic_modeling(self, questions: List[str], n_topics: int = 10):
        """
        Perform topic modeling on questions to identify misconception themes.
        
        Args:
            questions: List of question texts
            n_topics: Number of topics
            
        Returns:
            Trained BERTopic model or None if not available
        """
        if not BERTOPIC_AVAILABLE:
            print("BERTopic not available, skipping topic modeling")
            return None
        
        topic_model = BERTopic(nr_topics=n_topics, verbose=False)
        topics, probs = topic_model.fit_transform(questions)
        
        return topic_model
    
    def analyze_low_competency_questions(
        self, 
        df: pd.DataFrame, 
        competency_threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Analyze questions with low competency scores to identify misconceptions.
        
        Args:
            df: DataFrame with questions and competency scores
            competency_threshold: Threshold for low competency
            
        Returns:
            DataFrame with misconception analysis
        """
        df = df.copy()
        df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
        
        # Identify low competency questions
        low_comp_mask = df['competency_score'] < competency_threshold
        low_comp_df = df[low_comp_mask].copy()
        
        # Analyze each question
        analyses = []
        for idx, row in low_comp_df.iterrows():
            analysis = self.analyze_question_intent(row.get('question_text', ''))
            analyses.append(analysis)
        
        # Add analysis results to dataframe
        for i, analysis in enumerate(analyses):
            idx = low_comp_df.index[i]
            for key, value in analysis['indicators'].items():
                df.loc[idx, key] = value
            df.loc[idx, 'num_concepts'] = len(analysis['concepts'])
            df.loc[idx, 'concepts'] = ', '.join(list(analysis['concepts'])[:10])  # Store first 10
        
        return df
    
    def identify_misconception_patterns(
        self, 
        df: pd.DataFrame,
        competency_threshold: float = 0.3
    ) -> Dict[str, any]:
        """
        Identify common misconception patterns.
        
        Args:
            df: DataFrame with questions and competency scores
            competency_threshold: Threshold for low competency
            
        Returns:
            Dictionary with misconception patterns
        """
        df = df.copy()
        df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
        
        # Get low competency questions
        low_comp_df = df[df['competency_score'] < competency_threshold]
        
        if len(low_comp_df) == 0:
            return {}
        
        # Extract concepts from low competency questions
        all_concepts = []
        for _, row in low_comp_df.iterrows():
            concepts = self.concept_extractor.extract_concepts(row.get('question_text', ''))
            all_concepts.extend(concepts)
        
        # Count concept frequencies
        concept_counts = Counter(all_concepts)
        
        # Group by subject/chapter/bloom
        patterns = {}
        for col in ['Subject', 'chapter', 'Bloom_tag']:
            if col in low_comp_df.columns:
                patterns[f'{col}_distribution'] = low_comp_df[col].value_counts().to_dict()
        
        return {
            'common_concepts': dict(concept_counts.most_common(20)),
            'patterns': patterns,
            'total_low_comp_questions': len(low_comp_df)
        }
