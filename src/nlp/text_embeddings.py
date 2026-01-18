"""Text embedding utilities for NLP analysis."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import torch
import yaml
import logging

logger = logging.getLogger(__name__)

# Try to import LLM components
try:
    from ..llm.client import LLMClient
    from ..llm.cache import LLMCache
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMClient = None
    LLMCache = None


class TextEmbedder:
    """Generate embeddings for text using sentence transformers with optional LLM backend."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", config_path: str = "config.yaml"):
        """
        Initialize text embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            config_path: Path to configuration file
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Initialize LLM support if available
        self._llm_enabled = False
        self._llm_client = None
        self._llm_cache = None
        
        if LLM_AVAILABLE:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                llm_config = config.get('llm', {})
                if llm_config.get('use_llm_embeddings', False):
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
                        logger.info("LLM embeddings enabled")
                    else:
                        logger.info("LLM API not available, using local embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM embeddings: {e}, using local embeddings")
    
    def _llm_embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Generate embeddings using LLM API (internal method).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        if not self._llm_enabled or not self._llm_client:
            return None
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache first
        if self._llm_cache:
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = f"embed:{text}"
                cached = self._llm_cache.get(cache_key)
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
            embeddings = self._llm_client.embed(uncached_texts, batch_size=batch_size)
            
            if embeddings is None:
                return None
            
            # Cache new embeddings
            if self._llm_cache:
                for text, embedding in zip(uncached_texts, embeddings):
                    cache_key = f"embed:{text}"
                    self._llm_cache.set(cache_key, embedding.tolist())
            
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
        
        # No cache, direct LLM call
        return self._llm_client.embed(texts, batch_size=batch_size)
    
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
        
        # Try LLM first if enabled
        if self._llm_enabled:
            try:
                llm_embeddings = self._llm_embed(texts, batch_size=batch_size)
                if llm_embeddings is not None:
                    return llm_embeddings
            except Exception as e:
                logger.warning(f"LLM embedding failed, falling back to local: {e}")
        
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
