"""Caching layer for Claude responses."""

import hashlib
import time
import logging
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ClaudeCache:
    """LRU cache for Claude responses with TTL."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize Claude cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        hash_key = self._hash_key(key)
        
        if hash_key not in self.cache:
            return None
        
        value, timestamp = self.cache[hash_key]
        
        # Check if expired
        if time.time() - timestamp > self.ttl:
            del self.cache[hash_key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(hash_key)
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        hash_key = self._hash_key(key)
        timestamp = time.time()
        
        # Remove if exists
        if hash_key in self.cache:
            del self.cache[hash_key]
        
        # Add new entry
        self.cache[hash_key] = (value, timestamp)
        
        # Evict oldest if over limit
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
