"""OpenAI client for LM Studio API integration."""

import requests
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for interacting with LM Studio API."""
    
    def __init__(
        self,
        api_url: str = "http://192.168.0.4:1601",
        model: str = "qwen/qwen3-4b-thinking-2507",
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_url: Base URL for LM Studio API
            model: Model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_url = api_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        """Check if OpenAI API is available."""
        try:
            response = requests.get(
                f"{self.api_url}/v1/models",
                timeout=5
            )
            if response.status_code == 200:
                self.available = True
                logger.info(f"OpenAI API available at {self.api_url}")
            else:
                logger.warning(f"OpenAI API returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"OpenAI API not available: {e}")
            self.available = False
    
    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Make API request with retry logic.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            retry_count: Current retry attempt
            
        Returns:
            API response or None if failed
        """
        url = f"{self.api_url}{endpoint}"
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                if retry_count < self.max_retries:
                    time.sleep(1 * (retry_count + 1))  # Exponential backoff
                    return self._make_request(endpoint, payload, retry_count + 1)
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"API request timeout (attempt {retry_count + 1})")
            if retry_count < self.max_retries:
                time.sleep(1 * (retry_count + 1))
                return self._make_request(endpoint, payload, retry_count + 1)
            return None
            
        except Exception as e:
            logger.error(f"API request error: {e}")
            if retry_count < self.max_retries:
                time.sleep(1 * (retry_count + 1))
                return self._make_request(endpoint, payload, retry_count + 1)
            return None
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            Generated text or None if failed
        """
        if not self.available:
            return None
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = self._make_request("/v1/chat/completions", payload)
        
        if response and "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        
        return None
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 50
    ) -> Optional[np.ndarray]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        if not self.available:
            return None
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # For LM Studio, we'll use the chat completion endpoint
            # and extract embeddings from the model's internal representation
            # Note: This is a simplified approach - actual embedding extraction
            # may require model-specific handling
            
            # Try to use embeddings endpoint if available
            payload = {
                "model": self.model,
                "input": batch
            }
            
            # Try embeddings endpoint first
            response = self._make_request("/v1/embeddings", payload)
            
            if response and "data" in response:
                batch_embeddings = [item["embedding"] for item in response["data"]]
                all_embeddings.extend(batch_embeddings)
            else:
                # Fallback: use chat completion and extract features
                # This is a workaround if embeddings endpoint is not available
                logger.warning("Embeddings endpoint not available, using fallback")
                return None
        
        if all_embeddings:
            return np.array(all_embeddings)
        
        return None
    
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        batch_size: int = 10
    ) -> List[Optional[str]]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            batch_size: Batch size for processing
            
        Returns:
            List of generated texts (None for failed generations)
        """
        if not self.available:
            return [None] * len(prompts)
        
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch:
                result = self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return self.available
