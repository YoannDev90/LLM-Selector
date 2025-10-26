import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from llm_selector import LLMSelector, LLMSelectorConfig


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.fixture
    def sample_models(self):
        return {
            "gpt-4": {"description": "GPT-4 model", "weight": 2.0},
            "claude-3": {"description": "Claude 3 model", "weight": 1.0}
        }

    @pytest.fixture
    def sample_configs(self):
        return [
            LLMSelectorConfig("gpt-4", api_key="test-key"),
            LLMSelectorConfig("claude-3", api_key="test-key2")
        ]

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_api_error_fallback(self, mock_acompletion, sample_models, sample_configs):
        """Test fallback when all selector configs fail"""
        mock_acompletion.side_effect = Exception("API Error")

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        # Should fallback to default model
        assert result == "gpt-4"

        # Check metrics
        metrics = selector.get_metrics()
        assert metrics["failed_selections"] == 1
        assert metrics["successful_selections"] == 0

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_partial_api_failure(self, mock_acompletion, sample_models, sample_configs):
        """Test when first config fails but second succeeds"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Use claude-3"

        # First call fails, second succeeds
        mock_acompletion.side_effect = [Exception("First API Error"), mock_response]

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        assert result == "claude-3"
        assert mock_acompletion.call_count == 2

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_invalid_response_format(self, mock_acompletion, sample_models, sample_configs):
        """Test handling of invalid API response formats"""
        # Response without choices
        mock_response = MagicMock()
        mock_response.choices = None
        mock_acompletion.return_value = mock_response

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        # Should fallback to default
        assert result == "gpt-4"

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_empty_response_content(self, mock_acompletion, sample_models, sample_configs):
        """Test handling of empty response content"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_acompletion.return_value = mock_response

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        # Should fallback to default
        assert result == "gpt-4"

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_none_response_content(self, mock_acompletion, sample_models, sample_configs):
        """Test handling of None response content"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_acompletion.return_value = mock_response

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        # Should fallback to default
        assert result == "gpt-4"

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_malformed_choices(self, mock_acompletion, sample_models, sample_configs):
        """Test handling of malformed choices array"""
        mock_response = MagicMock()
        mock_response.choices = []  # Empty choices
        mock_acompletion.return_value = mock_response

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        # Should fallback to default
        assert result == "gpt-4"

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_malformed_message(self, mock_acompletion, sample_models, sample_configs):
        """Test handling of malformed message object"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = None  # No message
        mock_acompletion.return_value = mock_response

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        # Should fallback to default
        assert result == "gpt-4"

    def test_parse_selection_edge_cases(self, sample_models):
        """Test edge cases in selection parsing"""
        selector = LLMSelector(sample_models)

        # Test with None input
        assert selector.parse_selection(None) == "gpt-4"

        # Test with very long response
        long_response = "Use " + "gpt-4 " * 1000
        assert selector.parse_selection(long_response) == "gpt-4"

        # Test with special characters
        assert selector.parse_selection("Use gpt-4!@#$%^&*()") == "gpt-4"

        # Test with unicode characters
        assert selector.parse_selection("Use gpt-4 🌟🚀") == "gpt-4"

    def test_parse_selection_regex_edge_cases(self, sample_models):
        """Test regex patterns in selection parsing"""
        selector = LLMSelector(sample_models)

        # Test various recommendation patterns
        assert selector.parse_selection("I suggest gpt-4") == "gpt-4"
        assert selector.parse_selection("Try claude-3") == "claude-3"  # Should match claude-3
        assert selector.parse_selection("recommend gpt-4") == "gpt-4"
        assert selector.parse_selection("choose gpt-4") == "gpt-4"
        assert selector.parse_selection("select gpt-4") == "gpt-4"
        assert selector.parse_selection("pick gpt-4") == "gpt-4"
        assert selector.parse_selection("use gpt-4") == "gpt-4"
        assert selector.parse_selection("go with gpt-4") == "gpt-4"

    def test_parse_selection_case_insensitive(self, sample_models):
        """Test case insensitive parsing"""
        selector = LLMSelector(sample_models)

        assert selector.parse_selection("USE GPT-4") == "gpt-4"
        assert selector.parse_selection("use Gpt-4") == "gpt-4"
        assert selector.parse_selection("Use CLAUDE-3") == "claude-3"

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_network_timeout_simulation(self, mock_acompletion, sample_models, sample_configs):
        """Test handling of network timeouts"""
        mock_acompletion.side_effect = asyncio.TimeoutError("Request timeout")

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        # Should fallback to default
        assert result == "gpt-4"

        metrics = selector.get_metrics()
        assert metrics["failed_selections"] == 1

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_authentication_error(self, mock_acompletion, sample_models, sample_configs):
        """Test handling of authentication errors"""
        mock_acompletion.side_effect = Exception("Authentication failed")

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        # Should fallback to default
        assert result == "gpt-4"

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_rate_limit_error(self, mock_acompletion, sample_models, sample_configs):
        """Test handling of rate limit errors"""
        mock_acompletion.side_effect = Exception("Rate limit exceeded")

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")

        # Should fallback to default
        assert result == "gpt-4"

    def test_cache_edge_cases(self, sample_models):
        """Test cache edge cases"""
        selector = LLMSelector(sample_models, cache_enabled=True)

        # Test cache with None key
        selector._set_cache_result(None, "result")
        assert selector._get_cached_result(None) is None

        # Test cache with empty key
        selector._set_cache_result("", "result")
        assert selector._get_cached_result("") == "result"

        # Test cache expiration
        selector._set_cache_result("key", "result", ttl=0)  # Immediate expiration
        time.sleep(0.01)  # Small delay
        assert selector._get_cached_result("key") is None

    def test_metrics_edge_cases(self, sample_models):
        """Test metrics edge cases"""
        selector = LLMSelector(sample_models)

        # Test metrics with no requests
        metrics = selector.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["cache_hit_rate"] == 0.0

        # Test metrics reset
        selector.reset_metrics()
        reset_metrics = selector.get_metrics()
        assert all(v == 0 or v == 0.0 for v in reset_metrics.values() if isinstance(v, (int, float)))

    def test_cost_edge_cases(self, sample_models):
        """Test cost calculation edge cases"""
        selector = LLMSelector(sample_models)

        # Test cost estimation with invalid response
        cost = selector._estimate_cost("gpt-4", None)
        assert cost == 0.0

        cost = selector._estimate_cost("gpt-4", {})
        assert cost == 0.0

        cost = selector._estimate_cost("gpt-4", {"usage": None})
        assert cost == 0.0

        cost = selector._estimate_cost("gpt-4", {"usage": {}})
        assert cost == 0.0

        # Test with negative tokens (shouldn't happen but test robustness)
        cost = selector._estimate_cost("gpt-4", {"usage": {"total_tokens": -100}})
        assert cost == 0.0

    def test_budget_edge_cases(self, sample_models):
        """Test budget limit edge cases"""
        selector = LLMSelector(sample_models)

        # Test with zero budget
        selector.set_budget_limit(0.0)
        assert selector.is_budget_exceeded() is True

        # Test with negative budget
        selector.set_budget_limit(-1.0)
        assert selector.is_budget_exceeded() is True

        # Test with no budget limit
        selector.set_budget_limit(None)
        assert selector.is_budget_exceeded() is False

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")
    async def test_concurrent_requests(self, mock_acompletion, sample_models, sample_configs):
        """Test concurrent requests handling"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Use gpt-4"
        mock_acompletion.return_value = mock_response

        selector = LLMSelector(sample_models, selector_configs=sample_configs, cache_enabled=False)

        # Run multiple concurrent requests
        tasks = [selector.select("test query") for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r == "gpt-4" for r in results)
        assert mock_acompletion.call_count == 10  # No caching in this test

    def test_memory_cleanup(self, sample_models):
        """Test memory cleanup operations"""
        selector = LLMSelector(sample_models, cache_enabled=True)

        # Fill cache
        for i in range(100):
            selector._set_cache_result(f"key_{i}", f"result_{i}")

        initial_size = len(selector._cache)
        assert initial_size == 100

        # Clear cache
        selector.clear_cache()
        assert len(selector._cache) == 0

        # Reset metrics
        selector.metrics["total_requests"] = 100
        selector.reset_metrics()
        assert selector.metrics["total_requests"] == 0


class TestConfigurationErrors:
    """Test configuration-related errors"""

    def test_invalid_model_config(self):
        """Test invalid model configurations"""
        with pytest.raises(ValueError, match="models must be a non-empty dictionary"):
            LLMSelector({})

        with pytest.raises(ValueError, match="must have a 'description' field"):
            LLMSelector({"gpt-4": {}})

        with pytest.raises(ValueError, match="weight must be a number"):
            LLMSelector({"gpt-4": {"description": "test", "weight": "invalid"}})

        with pytest.raises(ValueError, match="weight must be positive"):
            LLMSelector({"gpt-4": {"description": "test", "weight": -1}})

    def test_invalid_selector_configs(self):
        """Test invalid selector configurations"""
        models = {"gpt-4": {"description": "test"}}

        # Empty configs should work (no default config added)
        selector = LLMSelector(models, selector_configs=[])
        assert len(selector.selector_configs) == 0

        # Invalid config type
        with pytest.raises(ValueError, match="selector_configs must be a list"):
            LLMSelector(models, selector_configs="invalid")

    def test_invalid_cache_config(self):
        """Test invalid cache configurations"""
        models = {"gpt-4": {"description": "test"}}

        # Invalid cache size
        with pytest.raises(ValueError, match="max_cache_size must be positive"):
            LLMSelector(models, max_cache_size=0)

        with pytest.raises(ValueError, match="max_cache_size must be positive"):
            LLMSelector(models, max_cache_size=-1)

        # Invalid cache TTL
        with pytest.raises(ValueError, match="cache_ttl must be non-negative"):
            LLMSelector(models, cache_ttl=-1)

    def test_invalid_retry_config(self):
        """Test invalid retry configurations"""
        models = {"gpt-4": {"description": "test"}}

        # Invalid max retries
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            LLMSelector(models, max_retries=-1)

        # Invalid retry delay
        with pytest.raises(ValueError, match="retry_delay must be positive"):
            LLMSelector(models, retry_delay=0)

        with pytest.raises(ValueError, match="retry_delay must be positive"):
            LLMSelector(models, retry_delay=-1)


if __name__ == "__main__":
    pytest.main([__file__])