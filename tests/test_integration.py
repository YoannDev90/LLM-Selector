import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
import tempfile
import os
from llm_selector import LLMSelector, LLMSelectorConfig


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complex scenarios"""

    @pytest.fixture
    def complex_models(self):
        """Complex model configuration for testing"""
        return {
            "gpt-4-turbo": {
                "description": "GPT-4 Turbo - Latest GPT-4 model with improved performance",
                "weight": 3.0,
                "cost_per_token": 0.00003,
                "context_window": 128000,
                "capabilities": ["text", "code", "analysis"]
            },
            "claude-3-opus": {
                "description": "Claude 3 Opus - Most capable Claude model",
                "weight": 2.5,
                "cost_per_token": 0.000015,
                "context_window": 200000,
                "capabilities": ["text", "code", "vision"]
            },
            "claude-3-sonnet": {
                "description": "Claude 3 Sonnet - Balanced performance and cost",
                "weight": 2.0,
                "cost_per_token": 0.000008,
                "context_window": 200000,
                "capabilities": ["text", "code"]
            },
            "gpt-3.5-turbo": {
                "description": "GPT-3.5 Turbo - Fast and cost-effective",
                "weight": 1.0,
                "cost_per_token": 0.000002,
                "context_window": 16385,
                "capabilities": ["text"]
            }
        }

    @pytest.fixture
    def multi_provider_configs(self):
        """Multiple provider configurations"""
        return [
            LLMSelectorConfig("openai/gpt-4-turbo", api_key="sk-openai-key"),
            LLMSelectorConfig("anthropic/claude-3-opus", api_key="sk-anthropic-key"),
            LLMSelectorConfig("openai/gpt-3.5-turbo", api_key="sk-openai-key")
        ]

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_model_selection_based_on_task_complexity(self, mock_acompletion, complex_models, multi_provider_configs):
        """Test that model selection adapts to task complexity"""
        # Mock responses for different complexity levels
        responses = {
            "simple": MagicMock(choices=[MagicMock(message=MagicMock(content="For this simple task, I recommend gpt-3.5-turbo"))]),
            "medium": MagicMock(choices=[MagicMock(message=MagicMock(content="For this balanced task, I recommend claude-3-sonnet"))]),
            "complex": MagicMock(choices=[MagicMock(message=MagicMock(content="For this complex analysis task, I recommend gpt-4-turbo"))]),
            "vision": MagicMock(choices=[MagicMock(message=MagicMock(content="For this vision task, I recommend claude-3-opus"))])
        }

        mock_acompletion.side_effect = [
            responses["simple"],   # Simple task
            responses["medium"],   # Medium task
            responses["complex"],  # Complex task
            responses["vision"]    # Vision task
        ]

        selector = LLMSelector(complex_models, selector_configs=multi_provider_configs)

        # Test different task complexities
        result1 = await selector.select("Write a simple hello world program")
        result2 = await selector.select("Analyze this business report and provide insights")
        result3 = await selector.select("Perform complex mathematical analysis on this dataset")
        result4 = await selector.select("Describe what you see in this image")

        assert result1 == "gpt-3.5-turbo"
        assert result2 == "claude-3-sonnet"
        assert result3 == "gpt-4-turbo"
        assert result4 == "claude-3-opus"

        assert mock_acompletion.call_count == 4

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_cost_optimization_scenario(self, mock_acompletion, complex_models):
        """Test cost optimization with budget constraints"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Use gpt-3.5-turbo"
        mock_response.usage = {"total_tokens": 100}
        mock_acompletion.return_value = mock_response

        # Add a default selector config for the test
        default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
        selector = LLMSelector(complex_models, selector_configs=[default_config], cache_enabled=True)
        selector.set_budget_limit(0.001)  # Very low budget

        # First request should work
        result1 = await selector.select("Simple task")
        assert result1 == "gpt-3.5-turbo"

        # Check cost tracking
        cost_after_first = selector.get_total_cost()
        assert cost_after_first > 0

        # Second request should still work (within budget)
        result2 = await selector.select("Another simple task")
        assert result2 == "gpt-3.5-turbo"

        # Verify caching worked
        metrics = selector.get_metrics()
        assert metrics["cache_hits"] == 0  # Different queries, no cache hit

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_provider_fallback_scenario(self, mock_acompletion, complex_models, multi_provider_configs):
        """Test fallback between different providers"""
        # First provider fails, second succeeds
        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "Use claude-3-opus"

        mock_acompletion.side_effect = [
            Exception("OpenAI API rate limit"),  # First provider fails
            success_response  # Second provider succeeds
        ]

        selector = LLMSelector(complex_models, selector_configs=multi_provider_configs)

        result = await selector.select("Complex analysis task")
        assert result == "claude-3-opus"
        assert mock_acompletion.call_count == 2

        # Check metrics
        metrics = selector.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_selections"] == 1
        assert metrics["failed_selections"] == 0  # Overall success

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, mock_acompletion, complex_models):
        """Test performance monitoring and metrics"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Use gpt-4-turbo"
        mock_response.usage = {"total_tokens": 150}
        mock_acompletion.return_value = mock_response

        # Add a default selector config for the test
        default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
        selector = LLMSelector(complex_models, selector_configs=[default_config], debug=True)

        start_time = time.time()
        result = await selector.select("Performance test query")
        end_time = time.time()

        assert result == "gpt-4-turbo"

        metrics = selector.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_selections"] == 1
        assert "average_latency" in metrics
        assert metrics["average_latency"] > 0
        assert metrics["total_cost"] > 0

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_cache_performance_optimization(self, mock_acompletion, complex_models):
        """Test cache performance optimization"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Use gpt-4-turbo"
        mock_response.usage = {"total_tokens": 200}
        mock_acompletion.return_value = mock_response

        # Add a default selector config for the test
        default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
        selector = LLMSelector(complex_models, selector_configs=[default_config], cache_enabled=True, max_cache_size=100)

        # First set of requests
        for i in range(5):
            result = await selector.select(f"Query {i}")
            assert result == "gpt-4-turbo"

        # Repeat the same queries (should use cache)
        for i in range(5):
            result = await selector.select(f"Query {i}")
            assert result == "gpt-4-turbo"

        # Check metrics
        metrics = selector.get_metrics()
        assert metrics["total_requests"] == 10
        assert metrics["cache_hits"] == 5
        assert metrics["cache_hit_rate"] == 0.5

        # Only 5 API calls should have been made (not 10)
        assert mock_acompletion.call_count == 5

    @patch("asyncio.sleep")
    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, mock_acompletion, mock_sleep, complex_models):
        """Test retry logic with exponential backoff"""
        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "Use gpt-4-turbo"

        # Fail twice, then succeed
        mock_acompletion.side_effect = [
            Exception("Temporary failure 1"),
            Exception("Temporary failure 2"),
            success_response
        ]

        # Add a default selector config for the test
        default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
        selector = LLMSelector(complex_models, selector_configs=[default_config], max_retries=3, retry_delay=0.1)

        result = await selector.select("Test query")
        assert result == "gpt-4-turbo"
        assert mock_acompletion.call_count == 3

        # Check exponential backoff timing
        expected_delays = [0.1, 0.2]  # Exponential backoff
        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        for expected, actual in zip(expected_delays, actual_delays):
            assert abs(actual - expected) < 0.01


    @pytest.mark.asyncio
    async def test_concurrent_load_test(self, complex_models):
        """Test concurrent load handling"""
        async def mock_completion(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate API delay
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Use gpt-4-turbo"
            response.usage = {"total_tokens": 100}
            return response

        with patch("llm_selector.selector.litellm.acompletion", side_effect=mock_completion):
            # Add a default selector config for the test
            default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
            selector = LLMSelector(complex_models, selector_configs=[default_config])

            # Run 20 concurrent requests
            tasks = [selector.select(f"Concurrent query {i}") for i in range(20)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(r == "gpt-4-turbo" for r in results)

            metrics = selector.get_metrics()
            assert metrics["total_requests"] == 20
            assert metrics["successful_selections"] == 20

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_memory_efficiency_with_large_cache(self, mock_acompletion, complex_models):
        """Test memory efficiency with large cache"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Use gpt-4-turbo"
        mock_acompletion.return_value = mock_response

        # Add a default selector config for the test
        default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
        selector = LLMSelector(complex_models, selector_configs=[default_config], cache_enabled=True, max_cache_size=50)

        # Fill cache with many unique queries
        for i in range(100):  # More than cache size
            result = await selector.select(f"Unique query {i}")
            assert result == "gpt-4-turbo"

        # Cache should be limited to max_cache_size
        assert len(selector._cache) <= 50

        # Check that cache eviction worked
        metrics = selector.get_metrics()
        assert metrics["total_requests"] == 100

    def test_configuration_validation_comprehensive(self, complex_models):
        """Test comprehensive configuration validation"""
        # Valid configuration
        selector = LLMSelector(complex_models)
        assert selector.validate_configuration() is True

        # Test with budget
        selector.set_budget_limit(100.0)
        assert selector.validate_configuration() is True

        # Test with cache settings
        selector = LLMSelector(complex_models, cache_enabled=True, max_cache_size=1000, cache_ttl=3600)
        assert selector.validate_configuration() is True

        # Test with retry settings
        selector = LLMSelector(complex_models, max_retries=5, retry_delay=1.0)
        assert selector.validate_configuration() is True

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_error_recovery_and_logging(self, mock_acompletion, complex_models, caplog):
        """Test error recovery and logging"""
        # Setup logging capture
        import logging
        caplog.set_level(logging.DEBUG)

        # First fail, then succeed
        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "Use claude-3-sonnet"

        mock_acompletion.side_effect = [
            Exception("Network timeout"),
            success_response
        ]

        # Add a default selector config for the test
        default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
        selector = LLMSelector(complex_models, selector_configs=[default_config], debug=True, max_retries=1)

        result = await selector.select("Test error recovery")
        assert result == "claude-3-sonnet"

        # Check that debug logs were generated
        assert "Attempt 1 failed" in caplog.text
        # Verify the selector recovered from the error
        assert result == "claude-3-sonnet"

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_cost_tracking_accuracy(self, mock_acompletion, complex_models):
        """Test cost tracking accuracy across multiple requests"""
        responses = []
        for i in range(3):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = f"Use gpt-4-turbo {i}"
            response.usage = {"total_tokens": 100 + i * 50}  # Different token counts
            responses.append(response)

        mock_acompletion.side_effect = responses

        # Add a default selector config for the test
        default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
        selector = LLMSelector(complex_models, selector_configs=[default_config])

        # Make three requests
        for i in range(3):
            result = await selector.select(f"Query {i}")
            assert result == "gpt-4-turbo"

        # Calculate expected total cost
        expected_cost = sum((100 + i * 50) * 0.00003 for i in range(3))

        metrics = selector.get_metrics()
        assert abs(metrics["total_cost"] - expected_cost) < 0.000001


@pytest.mark.integration
@pytest.mark.slow
class TestStressTests:
    """Stress tests for high load scenarios"""

    @pytest.fixture
    def stress_models(self):
        """Simplified models for stress testing"""
        return {
            "gpt-4": {"description": "GPT-4 model", "weight": 1.0, "cost_per_token": 0.00003},
            "claude-3": {"description": "Claude 3 model", "weight": 1.0, "cost_per_token": 0.000015}
        }

    async def mock_stress_completion(self, *args, **kwargs):
        """Mock completion for stress testing"""
        await asyncio.sleep(0.001)  # Minimal delay
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Use gpt-4"
        response.usage = {"total_tokens": 50}
        return response

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, mock_acompletion, stress_models):
        """Test high concurrency stress"""
        mock_acompletion.side_effect = self.mock_stress_completion

        # Add a default selector config for the test
        default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
        selector = LLMSelector(stress_models, selector_configs=[default_config])

        # Run 100 concurrent requests
        num_requests = 100
        tasks = [selector.select(f"Stress query {i}") for i in range(num_requests)]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # All should succeed
        assert len(results) == num_requests
        assert all(r == "gpt-4" for r in results)

        # Check performance
        total_time = end_time - start_time
        avg_time_per_request = total_time / num_requests

        metrics = selector.get_metrics()
        assert metrics["total_requests"] == num_requests
        assert metrics["successful_selections"] == num_requests

        # Performance should be reasonable (less than 1 second per request on average)
        assert avg_time_per_request < 1.0

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, mock_acompletion, stress_models):
        """Test prevention of memory leaks under load"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Use gpt-4"
        mock_acompletion.return_value = mock_response

        # Add a default selector config for the test
        default_config = LLMSelectorConfig("gpt-4", api_key="test-key")
        selector = LLMSelector(stress_models, selector_configs=[default_config], cache_enabled=True, max_cache_size=10)

        # Run many requests to test cache management
        for i in range(50):
            result = await selector.select(f"Memory test query {i}")
            assert result == "gpt-4"

        # Cache should not grow beyond limit
        assert len(selector._cache) <= 10

        # Force garbage collection check
        import gc
        gc.collect()

        # Selector should still work
        result = await selector.select("Final memory check")
        assert result == "gpt-4"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])