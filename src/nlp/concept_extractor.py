"""Concept extraction from question text."""

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Set, Dict, Optional
import pandas as pd
from collections import Counter
import json
import yaml
import logging

logger = logging.getLogger(__name__)

# Try to import Claude components
try:
    from ..claude.client import ClaudeClient
    from ..claude.cache import ClaudeCache
    from ..claude.prompts import get_concept_extraction_prompt
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    ClaudeClient = None
    ClaudeCache = None
    get_concept_extraction_prompt = None

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class ConceptExtractor:
    """Extract educational concepts from question text."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize concept extractor.
        
        Args:
            config_path: Path to configuration file
        """
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        # Initialize Claude support if available
        self._claude_enabled = False
        self._claude_client = None
        self._claude_cache = None
        
        if CLAUDE_AVAILABLE:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                claude_config = config.get('claude', {})
                if claude_config.get('use_claude_concepts', False):
                    self._claude_client = ClaudeClient(
                        api_url=claude_config.get('api_url', 'http://192.168.0.4:1601'),
                        model=claude_config.get('model', 'qwen/qwen3-4b-thinking-2507'),
                        timeout=claude_config.get('timeout', 30),
                        max_retries=claude_config.get('max_retries', 3)
                    )
                    
                    if self._claude_client.is_available():
                        self._claude_enabled = True
                        self._claude_cache = ClaudeCache(
                            max_size=claude_config.get('cache_max_size', 1000),
                            ttl=claude_config.get('cache_ttl', 3600)
                        )
                        logger.info("Claude concept extraction enabled")
                    else:
                        logger.info("Claude API not available, using local concept extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude concept extraction: {e}, using local extraction")
    
    def _claude_extract_concepts(self, text: str) -> Optional[Set[str]]:
        """
        Extract concepts using Claude API (internal method).
        
        Args:
            text: Input text
            
        Returns:
            Set of concepts or None if failed
        """
        if not self._claude_enabled or not self._claude_client or not get_concept_extraction_prompt:
            return None
        
        # Check cache first
        if self._claude_cache:
            cache_key = f"concepts:{text}"
            cached = self._claude_cache.get(cache_key)
            if cached is not None:
                return set(cached)
        
        try:
            prompt = get_concept_extraction_prompt(text)
            response = self._claude_client.generate(
                prompt,
                max_tokens=256,
                temperature=0.3
            )
            
            if response:
                # Try to parse JSON response
                try:
                    # Extract JSON from response (might have extra text)
                    json_start = response.find('[')
                    json_end = response.rfind(']') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        concepts_list = json.loads(json_str)
                        concepts = set(concepts_list)
                        
                        # Cache result
                        if self._claude_cache:
                            cache_key = f"concepts:{text}"
                            self._claude_cache.set(cache_key, list(concepts))
                        
                        return concepts
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract concepts from text
                    # Look for list-like patterns
                    lines = response.strip().split('\n')
                    concepts = set()
                    for line in lines:
                        line = line.strip()
                        # Remove common JSON array markers
                        line = line.lstrip('[').rstrip(']').rstrip(',').strip()
                        # Remove quotes
                        line = line.strip('"').strip("'")
                        if line and len(line) > 2:
                            concepts.add(line)
                    
                    if concepts:
                        # Cache result
                        if self._claude_cache:
                            cache_key = f"concepts:{text}"
                            self._claude_cache.set(cache_key, list(concepts))
                        
                        return concepts
        except Exception as e:
            logger.warning(f"Claude concept extraction failed: {e}")
        
        return None
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: tokenize, lemmatize, remove stopwords.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalnum() and token not in self.stop_words
        ]
        
        return tokens
    
    def extract_key_terms(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract key terms from text.
        
        Args:
            text: Input text
            min_length: Minimum length of terms
            
        Returns:
            List of key terms
        """
        tokens = self.preprocess_text(text)
        
        # Filter by length
        key_terms = [token for token in tokens if len(token) >= min_length]
        
        return key_terms
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases (simple pattern-based).
        
        Args:
            text: Input text
            
        Returns:
            List of noun phrases
        """
        # Simple pattern: sequences of capitalized words or common noun patterns
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized phrases
            r'\b\w+\s+(?:of|in|on|at|for|with)\s+\w+\b',  # Prepositional phrases
        ]
        
        noun_phrases = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            noun_phrases.extend(matches)
        
        return list(set(noun_phrases))
    
    def extract_concepts(self, text: str) -> Set[str]:
        """
        Extract educational concepts from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted concepts
        """
        # Try Claude first if enabled
        if self._claude_enabled:
            try:
                claude_concepts = self._claude_extract_concepts(text)
                if claude_concepts is not None and len(claude_concepts) > 0:
                    return claude_concepts
            except Exception as e:
                logger.warning(f"Claude concept extraction failed, falling back to local: {e}")
        
        # Fallback to local extraction (existing implementation)
        concepts = set()
        
        # Extract key terms
        key_terms = self.extract_key_terms(text)
        concepts.update(key_terms)
        
        # Extract noun phrases
        noun_phrases = self.extract_noun_phrases(text)
        concepts.update(noun_phrases)
        
        return concepts
    
    def extract_concepts_batch(self, texts: List[str]) -> List[Set[str]]:
        """
        Extract concepts from multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of concept sets
        """
        # If Claude enabled, can batch process, but for now use individual calls
        # to maintain same interface
        return [self.extract_concepts(text) for text in texts]
