import logging
import litellm
import os
import re
import asyncio
import time
from typing import Optional, Dict, Any
from .config import logger_name, LLM_SELECTOR_PROMPT

logger = logging.getLogger(logger_name)

class LLMModel:
    """
    Represents a Large Language Model configuration.
    
    Attributes:
        name (str): The model identifier/name
        description (str): Human-readable description of the model's capabilities
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class LLMSelectorConfig:
    """
    Configuration for an LLM selector service.
    
    This class handles the configuration of a specific LLM that will be used
    to perform model selection. It supports various API key sources and
    additional parameters for the underlying LLM service.
    
    Attributes:
        model (str): The model identifier (e.g., 'openai/gpt-4')
        api_base (str, optional): Custom API base URL
        api_key (str, optional): API key for authentication
        extra_kwargs (dict): Additional parameters passed to litellm
    """
    def __init__(self, model: str, api_base: str = None, api_key: str = None, **kwargs):
        if not model or not isinstance(model, str):
            raise ValueError("Model name must be a non-empty string")
        
        self.model = model
        self.api_base = api_base
        self.api_key = self._resolve_api_key(api_key)
        self.extra_kwargs = kwargs
    
    def _resolve_api_key(self, api_key):
        if not api_key:
            return None
        if api_key.startswith("env:"):
            env_var = api_key[4:]
            resolved = os.getenv(env_var)
            if not resolved:
                logger.warning(f"Environment variable {env_var} not found")
            return resolved
        elif api_key.startswith("dotenv:"):
            from dotenv import load_dotenv
            load_dotenv()
            dotenv_key = api_key[7:]
            resolved = os.getenv(dotenv_key)
            if not resolved:
                logger.warning(f"dotenv key {dotenv_key} not found")
            return resolved
        else:
            # Clé en clair
            return api_key
    
    def validate(self) -> bool:
        """
        Validate this configuration.
        
        Performs basic validation checks on the configuration parameters
        including model name format and URL validation.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        issues = []
        
        if not self.model:
            issues.append("Model name is required")
        
        # Check if model is supported by litellm
        try:
            # This is a basic check - litellm might support more
            supported_providers = ['openai', 'anthropic', 'cohere', 'replicate', 'huggingface', 'openrouter']
            provider = self.model.split('/')[0] if '/' in self.model else self.model
            if not any(supported in provider.lower() for supported in supported_providers):
                logger.warning(f"Model provider '{provider}' may not be supported by litellm")
        except:
            pass
            
        if self.api_base and not self.api_base.startswith(('http://', 'https://')):
            issues.append(f"api_base should be a valid URL: {self.api_base}")
            
        if issues:
            logger.warning(f"Configuration issues for {self.model}: {', '.join(issues)}")
            return False
            
        return True

class LLMSelector:
    """
    Intelligent LLM model selector using LLM-based decision making.
    
    This class provides an intelligent way to select the most appropriate LLM
    model for a given task by using another LLM to make the selection decision.
    It supports multiple models with different capabilities, costs, and weights,
    along with comprehensive caching, metrics, and error handling.
    
    Key Features:
    - LLM-powered model selection based on task requirements
    - Multi-provider support with automatic fallback
    - Intelligent caching with TTL and size limits
    - Cost tracking and budget management
    - Comprehensive metrics and monitoring
    - Retry logic with exponential backoff
    - Synchronous and asynchronous interfaces
    
    Attributes:
        models (dict): Dictionary of available models with their configurations
        default_model (str): Fallback model when selection fails
        selector_configs (list): List of LLMSelectorConfig for selection services
        metrics (dict): Performance and usage metrics
        cache_enabled (bool): Whether caching is enabled
        cache_ttl (int): Cache time-to-live in seconds
        max_retries (int): Maximum retry attempts for API calls
        retry_delay (float): Base delay between retries in seconds
        cost_per_model (dict): Cost per token for each model
        preprompt (str): System prompt for the selector LLM
        litellm_kwargs (dict): Additional kwargs for litellm calls
        debug (bool): Enable debug logging
        max_cache_size (int): Maximum cache entries
        _cache (dict): Internal cache storage
    """
    def __init__(self, models: dict, preprompt: str = None, selector_configs: list = None, default_model: str = None, validate_configs: bool = True, max_retries: int = 3, retry_delay: float = 1.0, cache_enabled: bool = True, cache_ttl: int = 3600, debug: bool = False, max_cache_size: int = 1000, **litellm_kwargs):
        if not models or not isinstance(models, dict):
            raise ValueError("models must be a non-empty dictionary")
        
        # Validate model configurations
        for model_name, model_config in models.items():
            if not isinstance(model_config, dict):
                raise ValueError(f"Model configuration for '{model_name}' must be a dictionary")
            if 'description' not in model_config:
                raise ValueError(f"Model '{model_name}' must have a 'description' field")
            if 'weight' in model_config and not isinstance(model_config['weight'], (int, float)):
                raise ValueError(f"Model '{model_name}' weight must be a number")
            if 'weight' in model_config and model_config['weight'] <= 0:
                raise ValueError(f"Model '{model_name}' weight must be positive")
        
        self.models = models
        self.default_model = default_model or next(iter(models.keys()))
        self.selector_configs = selector_configs or []
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'config_failures': {},
            'average_latency': 0.0,
            'last_request_time': None,
            'total_cost': 0.0,
            'cost_by_model': {}
        }
        self._cache = {}
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cost_per_model = {model: data.get('cost_per_token', 0.0) for model, data in models.items()}
        self.preprompt = preprompt or LLM_SELECTOR_PROMPT
        self.litellm_kwargs = litellm_kwargs
        
        self.debug = debug
        
        # Memory optimization - limit cache size
        self.max_cache_size = max_cache_size
        
        # Validate configuration parameters
        if self.max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")
        if self.cache_ttl < 0:
            raise ValueError("cache_ttl must be non-negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay <= 0:
            raise ValueError("retry_delay must be positive")
        if selector_configs is not None and not isinstance(selector_configs, list):
            raise ValueError("selector_configs must be a list")
        
        if validate_configs and self.selector_configs:
            self._validate_configs()
    
    def _debug_log(self, message: str, *args):
        """Log debug messages if debug mode is enabled"""
        if self.debug:
            logger.debug(message, *args)
    
    def _set_cache_result(self, cache_key: str, result: str, ttl: int = None):
        """Cache a result with size limit"""
        if self.cache_enabled and cache_key is not None:
            # Clean expired entries and enforce size limit
            current_time = time.time()
            valid_entries = {
                k: v for k, v in self._cache.items() 
                if current_time - v['timestamp'] < self.cache_ttl
            }
            
            if len(valid_entries) >= self.max_cache_size:
                # Remove oldest entries
                sorted_entries = sorted(valid_entries.items(), key=lambda x: x[1]['timestamp'])
                valid_entries = dict(sorted_entries[-self.max_cache_size + 1:])
            
            self._cache = valid_entries
            # Use provided TTL or default cache TTL
            cache_ttl = ttl if ttl is not None else self.cache_ttl
            self._cache[cache_key] = {
                'result': result,
                'timestamp': current_time - (self.cache_ttl - cache_ttl)  # Adjust timestamp for custom TTL
            }
            self._debug_log(f"Cached result for key {cache_key}")
    
    def clear_cache(self):
        """Clear all cached results and reset cache storage."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared ({cache_size} entries removed)")
    
    def _get_cache_key(self, input_text: str) -> str:
        """Generate a cache key from input text"""
        # Simple hash of the input for cache key
        return str(hash(input_text + str(sorted(self.models.items()))))
    
    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """Get cached result if valid"""
        if not self.cache_enabled:
            return None
            
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                self._debug_log(f"Cache hit for key {cache_key}")
                return cached['result']
            else:
                # Expired, remove from cache
                del self._cache[cache_key]
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance and usage metrics.
        
        Returns a comprehensive dictionary of metrics including request counts,
        cache statistics, latency information, and cost tracking.
        
        Returns:
            Dict[str, Any]: Dictionary containing all current metrics
        """
        metrics = self.metrics.copy()
        metrics['cache_size'] = len(self._cache)
        metrics['cache_hit_rate'] = (
            self.metrics['cache_hits'] / self.metrics['total_requests'] 
            if self.metrics['total_requests'] > 0 else 0
        )
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'config_failures': {},
            'average_latency': 0.0,
            'last_request_time': None,
            'total_cost': 0.0,
            'cost_by_model': {}
        }
    
    def _estimate_cost(self, model: str, response: dict) -> float:
        """Estimate cost of a request based on token usage"""
        if model not in self.cost_per_model or response is None:
            return 0.0
        
        cost_per_token = self.cost_per_model[model]
        
        # Try to get token usage from response
        # Handle both dict and object responses
        if isinstance(response, dict):
            usage = response.get('usage', {})
        else:
            # Handle object with attributes
            usage = getattr(response, 'usage', {})
        
        if usage is None:
            usage = {}
        
        # Handle both dict and object usage
        if isinstance(usage, dict):
            total_tokens = usage.get('total_tokens', usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0))
        else:
            # Handle object with attributes
            total_tokens = getattr(usage, 'total_tokens', getattr(usage, 'prompt_tokens', 0) + getattr(usage, 'completion_tokens', 0))
        
        if total_tokens == 0:
            # Estimate based on text length if no token info
            # Rough estimate: 1 token ≈ 4 characters
            if isinstance(response, dict):
                content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            else:
                choices = getattr(response, 'choices', [{}])
                if choices and len(choices) > 0:
                    message = getattr(choices[0], 'message', {})
                    content = getattr(message, 'content', '')
                else:
                    content = ''
            total_tokens = len(str(content)) // 4
        
        return total_tokens * cost_per_token
    
    def get_total_cost(self) -> float:
        """
        Get the total estimated cost of all API calls.
        
        Returns:
            float: Total cost in the currency unit used for cost_per_token
        """
        return self.metrics['total_cost']
    
    def set_budget_limit(self, limit: float):
        """
        Set a budget limit for API costs.
        
        Args:
            limit (float): Maximum allowed cost before is_budget_exceeded() returns True
        """
        self.budget_limit = limit
    
    def is_budget_exceeded(self) -> bool:
        """
        Check if the current total cost exceeds the budget limit.
        
        Returns:
            bool: True if budget is exceeded, False otherwise
        """
        return (hasattr(self, 'budget_limit') and 
                self.budget_limit is not None and 
                self.metrics['total_cost'] >= self.budget_limit)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry"""
        error_msg = str(error).lower()
        
        # Retry on network/timeout errors
        retryable_patterns = [
            'timeout', 'connection', 'network', 'rate limit', '429', 
            '502', '503', '504', 'server error', 'temporary failure'
        ]
        
        return any(pattern in error_msg for pattern in retryable_patterns)
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute a function with exponential backoff retry"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if attempt == self.max_retries or not self._is_retryable_error(e):
                    raise e
                
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        raise last_error
    
    def _validate_configs(self):
        """Validate all selector configurations"""
        try:
            # Check models
            if not self.models or not isinstance(self.models, dict):
                raise ValueError("models must be a non-empty dictionary")
            
            # Check default model exists
            if self.default_model not in self.models:
                raise ValueError("default_model must be in models")
            
            # Check cache settings
            if self.cache_ttl < 0:
                raise ValueError("cache_ttl must be non-negative")
            
            # Check retry settings
            if self.max_retries < 0:
                raise ValueError("max_retries must be non-negative")
            if self.retry_delay <= 0:
                raise ValueError("retry_delay must be positive")
            
            # Check selector configs if present
            for config in self.selector_configs:
                if not config.validate():
                    raise ValueError(f"Invalid selector config: {config.model}")
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")

    def validate_configuration(self) -> bool:
        """Validate the current configuration"""
        try:
            self._validate_configs()
            return True
        except Exception:
            return False

    async def select(self, input: str) -> str:
        """
        Select the most appropriate LLM model for the given input.
        
        Uses an LLM to analyze the input text and select the best model from
        the available options based on the configured selection criteria.
        Implements caching, retry logic, and fallback mechanisms.
        
        Args:
            input (str): The user input/query to analyze for model selection
            
        Returns:
            str: The selected model name
            
        Raises:
            Exception: If all selector configurations fail after retries
        """
        start_time = time.time()
        self.metrics['total_requests'] += 1
        self.metrics['last_request_time'] = start_time
        
        # Check cache first
        cache_key = self._get_cache_key(input)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.metrics['cache_hits'] += 1
            latency = time.time() - start_time
            self._update_latency(latency)
            return cached_result
        
        models_text = "\n".join(f"- {k} : {v.get('description', '')} (weight: {v.get('weight', 1.0)})" for k, v in self.models.items())
        
        messages = [
            {"role": "system", "content": self.preprompt + models_text},
            {"role": "user", "content": input}
        ]

        for config in self.selector_configs:
            try:
                # Use retry logic for each config attempt
                response = await self._retry_with_backoff(self._call_llm, config, messages)
                text = response.choices[0].message.content
                break
            except Exception as e:
                config_key = config.model
                if config_key not in self.metrics['config_failures']:
                    self.metrics['config_failures'][config_key] = 0
                self.metrics['config_failures'][config_key] += 1
                
                logger.error(f"LLM Selector config {config.model} failed after retries: {e}. Trying next config.")
                continue
        else:
            logger.error("All LLM Selector configs failed. Using default model.")
            self.metrics['failed_selections'] += 1
            result = self.default_model
            self._set_cache_result(cache_key, result)
            latency = time.time() - start_time
            self._update_latency(latency)
            return result
        
        result = self.parse_selection(text)
        self.metrics['successful_selections'] += 1
        
        # Track cost for the selected model
        if result in self.cost_per_model:
            estimated_cost = self._estimate_cost(result, response)
            self.metrics['total_cost'] += estimated_cost
            if result not in self.metrics['cost_by_model']:
                self.metrics['cost_by_model'][result] = 0.0
            self.metrics['cost_by_model'][result] += estimated_cost
        
        self._set_cache_result(cache_key, result)
        
        latency = time.time() - start_time
        self._update_latency(latency)
        return result
    
    def _update_latency(self, latency: float):
        """Update rolling average latency"""
        if self.metrics['average_latency'] == 0:
            self.metrics['average_latency'] = latency
        else:
            # Simple moving average
            self.metrics['average_latency'] = (
                self.metrics['average_latency'] * 0.9 + latency * 0.1
            )
    
    def select_sync(self, input: str) -> str:
        """
        Synchronous version of select() for convenience.
        
        Creates a new event loop to run the async select method.
        Warning: If called from within an async context, this may not work properly.
        
        Args:
            input (str): The user input/query to analyze
            
        Returns:
            str: The selected model name
            
        Raises:
            RuntimeError: If called from within an existing async context
            Exception: If all selector configurations fail after retries
        """
        try:
            # Try to run in new event loop
            return asyncio.run(self.select(input))
        except RuntimeError as e:
            if "already running" in str(e):
                raise RuntimeError("Cannot use select_sync() from within an async context. Use select() instead.")
            raise
    
    async def _call_llm(self, config: LLMSelectorConfig, messages: list) -> dict:
        """Make the actual LLM call for a specific config"""
        kwargs = {"model": config.model, "messages": messages}
        if config.api_base:
            kwargs["api_base"] = config.api_base
        if config.api_key:
            kwargs["api_key"] = config.api_key
        kwargs.update(config.extra_kwargs)
        kwargs.update(self.litellm_kwargs)
        
        response = await litellm.acompletion(**kwargs)
        return response

    def parse_selection(self, text: str) -> str:
        """
        Parse the LLM response to extract the selected model name.
        
        Uses multiple strategies for robustness:
        1. Exact model name matches
        2. Regex patterns for common response formats
        3. Fuzzy matching for partial matches
        
        Args:
            text (str): The raw response text from the selector LLM
            
        Returns:
            str: The parsed model name, or default_model if parsing fails
        """
        if not text or not text.strip():
            logger.warning("Empty response from LLM selector")
            return self.default_model
            
        text = text.strip().lower()
        
        # Strategy 1: Look for exact model name matches
        for model_name in self.models.keys():
            if model_name.lower() in text:
                return model_name
        
        # Strategy 2: Regex patterns for common response formats
        patterns = [
            r'(?:select|choose|use|pick|go with)\s*[:\-]?\s*([a-zA-Z0-9\-_/]+)',
            r'model\s*[:\-]?\s*([a-zA-Z0-9\-_/]+)',
            r'([a-zA-Z0-9\-_/]+)\s+(?:is|would be|should be)\s+(?:the|my|a)\s+(?:best|recommended|chosen)',
            r'^([a-zA-Z0-9\-_/]+)$',  # Just the model name
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Validate that it's a known model
                for model_name in self.models.keys():
                    if candidate in model_name.lower() or model_name.lower() in candidate:
                        return model_name
        
        # Strategy 3: Fuzzy matching - find closest model name
        words = re.findall(r'\b[a-zA-Z0-9\-_/]+\b', text)
        for word in words:
            for model_name in self.models.keys():
                # Check if word is part of model name or vice versa
                model_lower = model_name.lower()
                word_lower = word.lower()
                # Split model name by common separators
                model_parts = re.split(r'[-_/]', model_lower)
                word_parts = re.split(r'[-_/]', word_lower)
                
                # Check various matching conditions
                if (word_lower in model_lower or 
                    model_lower in word_lower or
                    any(part in word_lower for part in model_parts) or
                    any(part in model_lower for part in word_parts)):
                    return model_name
        
        logger.warning(f"Could not parse model from response: {text[:100]}...")
        return self.default_model