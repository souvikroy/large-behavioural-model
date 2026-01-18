"""Text embedding utilities for NLP analysis."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import torch
import yaml
import logging

logger = logging.getLogger(__name__)

# Try to import OpenAI components
try:
    from ..openai.client import OpenAIClient
    from ..openai.cache import OpenAICache
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIClient = None
    OpenAICache = None


class TextEmbedder:
    """Generate embeddings for text using sentence transformers with optional OpenAI backend."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", config_path: str = "config.yaml"):
        """
        Initialize text embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            config_path: Path to configuration file
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Initialize OpenAI support if available
        self._openai_enabled = False
        self._openai_client = None
        self._openai_cache = None
        
        if OPENAI_AVAILABLE:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                openai_config = config.get('openai', {})
                if openai_config.get('use_openai_embeddings', False):
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
                        logger.info("OpenAI embeddings enabled")
                    else:
                        logger.info("OpenAI API not available, using local embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embeddings: {e}, using local embeddings")
    
    def _openai_embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Generate embeddings using OpenAI API (internal method).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        if not self._openai_enabled or not self._openai_client:
            return None
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache first
        if self._openai_cache:
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = f"embed:{text}"
                cached = self._openai_cache.get(cache_key)
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
            embeddings = self._openai_client.embed(uncached_texts, batch_size=batch_size)
            
            if embeddings is None:
                return None
            
            # Cache new embeddings
            if self._openai_cache:
                for text, embedding in zip(uncached_texts, embeddings):
                    cache_key = f"embed:{text}"
                    self._openai_cache.set(cache_key, embedding.tolist())
            
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
        
        # No cache, direct OpenAI call
        return self._openai_client.embed(texts, batch_size=batch_size)
    
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
        
        # Try OpenAI first if enabled
        if self._openai_enabled:
            try:
                openai_embeddings = self._openai_embed(texts, batch_size=batch_size)
                if openai_embeddings is not None:
                    return openai_embeddings
            except Exception as e:
                logger.warning(f"OpenAI embedding failed, falling back to local: {e}")
        
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
