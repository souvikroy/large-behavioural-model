"""Text embedding utilities for NLP analysis."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import torch
import yaml
import logging

logger = logging.getLogger(__name__)

# Try to import Claude components
try:
    from ..claude.client import ClaudeClient
    from ..claude.cache import ClaudeCache
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    ClaudeClient = None
    ClaudeCache = None


class TextEmbedder:
    """Generate embeddings for text using sentence transformers with optional Claude backend."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", config_path: str = "config.yaml"):
        """
        Initialize text embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            config_path: Path to configuration file
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Initialize Claude support if available
        self._claude_enabled = False
        self._claude_client = None
        self._claude_cache = None
        
        if CLAUDE_AVAILABLE:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                claude_config = config.get('claude', {})
                if claude_config.get('use_claude_embeddings', False):
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
                        logger.info("Claude embeddings enabled")
                    else:
                        logger.info("Claude API not available, using local embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude embeddings: {e}, using local embeddings")
    
    def _claude_embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Generate embeddings using Claude API (internal method).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        if not self._claude_enabled or not self._claude_client:
            return None
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache first
        if self._claude_cache:
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = f"embed:{text}"
                cached = self._claude_cache.get(cache_key)
                if cached is not None:
                    cached_results.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # If all cached, return cached results
            if len(uncached_texts) == 0:
                result = np.array([cached_results[i][1] for i in range(len(texts))])
                return result
            
            # Get embeddings for uncached texts
            embeddings = self._claude_client.embed(uncached_texts, batch_size=batch_size)
            
            if embeddings is None:
                return None
            
            # Cache new embeddings
            if self._claude_cache:
                for text, embedding in zip(uncached_texts, embeddings):
                    cache_key = f"embed:{text}"
                    self._claude_cache.set(cache_key, embedding.tolist())
            
            # Combine cached and new embeddings
            if len(cached_results) > 0:
                result = np.zeros((len(texts), embeddings.shape[1]))
                cached_dict = {idx: emb for idx, emb in cached_results}
                
                for i in range(len(texts)):
                    if i in cached_dict:
                        result[i] = cached_dict[i]
                    else:
                        uncached_idx = uncached_indices.index(i)
                        result[i] = embeddings[uncached_idx]
                
                return result
            
            return embeddings
        
        # No cache, direct Claude call
        return self._claude_client.embed(texts, batch_size=batch_size)
    
    def embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Try Claude first if enabled
        if self._claude_enabled:
            try:
                claude_embeddings = self._claude_embed(texts, batch_size=batch_size)
                if claude_embeddings is not None:
                    return claude_embeddings
            except Exception as e:
                logger.warning(f"Claude embedding failed, falling back to local: {e}")
        
        # Fallback to local model (existing implementation)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_questions(self, questions: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for question texts.
        
        Args:
            questions: List of question texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of question embeddings
        """
        return self.embed(questions, batch_size=batch_size)
