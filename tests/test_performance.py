import pytest
import asyncio
import time
import statistics
from unittest.mock import AsyncMock, MagicMock, patch
from llm_selector import LLMSelector, LLMSelectorConfig


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks and timing tests"""

    @pytest.fixture
    def benchmark_models(self):
        """Models for performance testing"""
        return {
            "gpt-4": {"description": "GPT-4 model", "weight": 1.0},
            "claude-3": {"description": "Claude 3 model", "weight": 1.0},
            "llama-3": {"description": "Llama 3 model", "weight": 1.0}
        }

    async def mock_fast_completion(self, *args, **kwargs):
        """Fast mock completion for performance testing"""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Use gpt-4"
        response.usage = {"total_tokens": 50}
        return response

    async def mock_slow_completion(self, delay=0.1, *args, **kwargs):
        """Slow mock completion for performance testing"""
        await asyncio.sleep(delay)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Use gpt-4"
        response.usage = {"total_tokens": 50}
        return response

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_selection_latency_baseline(self, mock_acompletion, benchmark_models):
        """Test baseline selection latency"""
        mock_acompletion.side_effect = self.mock_fast_completion

        selector = LLMSelector(benchmark_models)

        # Warm up
        await selector.select("warmup")

        # Measure latency for multiple requests
        latencies = []
        num_samples = 10

        for _ in range(num_samples):
            start_time = time.perf_counter()
            result = await selector.select("test query")
            end_time = time.perf_counter()

            assert result == "gpt-4"
            latencies.append(end_time - start_time)

        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

        # Performance assertions (reasonable bounds)
        assert avg_latency < 0.1  # Less than 100ms average
        assert p95_latency < 0.2  # Less than 200ms p95

        # Update metrics
        metrics = selector.get_metrics()
        assert metrics["average_latency"] > 0

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_cache_performance_improvement(self, mock_acompletion, benchmark_models):
        """Test cache performance improvement"""
        mock_acompletion.side_effect = self.mock_fast_completion

        selector = LLMSelector(benchmark_models, cache_enabled=True)

        query = "cached query test"

        # First request (cache miss)
        start_time = time.perf_counter()
        result1 = await selector.select(query)
        first_request_time = time.perf_counter() - start_time

        # Second request (cache hit)
        start_time = time.perf_counter()
        result2 = await selector.select(query)
        second_request_time = time.perf_counter() - start_time

        assert result1 == "gpt-4"
        assert result2 == "gpt-4"

        # Cache hit should be significantly faster
        assert second_request_time < first_request_time * 0.5  # At least 2x faster

        # Check cache metrics
        metrics = selector.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_hit_rate"] == 0.5

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_concurrent_performance(self, mock_acompletion, benchmark_models):
        """Test concurrent request performance"""
        mock_acompletion.side_effect = self.mock_fast_completion

        selector = LLMSelector(benchmark_models)

        async def single_request(i):
            start_time = time.perf_counter()
            result = await selector.select(f"concurrent query {i}")
            end_time = time.perf_counter()
            return result, end_time - start_time

        # Test with different concurrency levels
        concurrency_levels = [1, 5, 10, 20]

        for concurrency in concurrency_levels:
            # Reset metrics
            selector.reset_metrics()

            # Run concurrent requests
            tasks = [single_request(i) for i in range(concurrency)]
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time

            # All should succeed
            assert len(results) == concurrency
            assert all(r[0] == "gpt-4" for r in results)

            # Calculate metrics
            latencies = [r[1] for r in results]
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)

            # Performance should scale reasonably
            # For low concurrency, should be close to sequential
            # For high concurrency, should show some parallelization benefit
            if concurrency == 1:
                baseline_latency = avg_latency
            else:
                # High concurrency should not be worse than 2x baseline per request
                assert avg_latency < baseline_latency * 2

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_memory_usage_with_cache(self, mock_acompletion, benchmark_models):
        """Test memory usage patterns with cache"""
        mock_acompletion.side_effect = self.mock_fast_completion

        selector = LLMSelector(benchmark_models, cache_enabled=True, max_cache_size=100)

        # Measure initial cache size
        initial_cache_size = len(selector._cache)

        # Add items to cache
        for i in range(50):
            await selector.select(f"memory test query {i}")

        mid_cache_size = len(selector._cache)
        assert mid_cache_size == 50

        # Add more items (should trigger eviction)
        for i in range(60):  # Total 110, exceeds max_cache_size
            await selector.select(f"memory test query overflow {i}")

        final_cache_size = len(selector._cache)
        assert final_cache_size <= 100  # Should not exceed max_cache_size

        # Cache should contain most recent items
        # (This is a simple check - in practice, we'd use a proper LRU cache)

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_retry_performance_impact(self, mock_acompletion, benchmark_models):
        """Test performance impact of retries"""
        # Setup: first call fails, second succeeds
        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "Use gpt-4"

        mock_acompletion.side_effect = [
            Exception("Temporary failure"),
            success_response
        ]

        selector = LLMSelector(benchmark_models, max_retries=1, retry_delay=0.05)

        start_time = time.perf_counter()
        result = await selector.select("retry test query")
        total_time = time.perf_counter() - start_time

        assert result == "gpt-4"

        # Should take at least retry_delay + some processing time
        assert total_time >= 0.05

        # But should not be excessive
        assert total_time < 1.0


    @pytest.mark.asyncio
    async def test_async_vs_sync_performance(self, benchmark_models):
        """Compare async vs sync performance"""
        # Note: This test is limited since we can't easily mock sync calls
        # in the same way. It's more of a structural test.

        selector = LLMSelector(benchmark_models)

        # Test that sync method raises error in async context
        async def test_sync_in_async():
            with pytest.raises(RuntimeError, match="Cannot use select_sync"):
                selector.select_sync("test")

        await test_sync_in_async()

        # Test that async method works
        with patch("llm_selector.selector.litellm.acompletion", side_effect=self.mock_fast_completion):
            result = await selector.select("async test")
            assert result == "gpt-4"

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_metrics_overhead(self, mock_acompletion, benchmark_models):
        """Test performance overhead of metrics collection"""
        mock_acompletion.side_effect = self.mock_fast_completion

        # Test with metrics enabled (default)
        selector_with_metrics = LLMSelector(benchmark_models)

        # Test with metrics disabled (if we had such an option)
        # For now, just measure the overhead of the current implementation

        latencies = []
        for _ in range(20):
            start_time = time.perf_counter()
            result = await selector_with_metrics.select("metrics overhead test")
            end_time = time.perf_counter()

            assert result == "gpt-4"
            latencies.append(end_time - start_time)

        avg_latency = statistics.mean(latencies)

        # Metrics overhead should be minimal
        assert avg_latency < 0.05  # Less than 50ms per request

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_scaling_performance(self, mock_acompletion, benchmark_models):
        """Test performance scaling with different model counts"""
        mock_acompletion.side_effect = self.mock_fast_completion

        # Test with different numbers of models
        model_counts = [1, 5, 10, 20]

        for count in model_counts:
            test_models = {
                f"model-{i}": {"description": f"Test model {i}", "weight": 1.0}
                for i in range(count)
            }

            selector = LLMSelector(test_models)

            # Measure selection time
            start_time = time.perf_counter()
            result = await selector.select("scaling test")
            end_time = time.perf_counter()

            latency = end_time - start_time

            # Performance should degrade gracefully with more models
            # but remain reasonable
            assert latency < 0.1  # Less than 100ms even with many models

            # Result should be one of the available models
            assert result in test_models


@pytest.mark.performance
@pytest.mark.slow
class TestLoadTests:
    """Load tests for high-throughput scenarios"""

    @pytest.fixture
    def load_test_models(self):
        """Simplified models for load testing"""
        return {
            "gpt-4": {"description": "GPT-4", "weight": 1.0, "cost_per_token": 0.00003}
        }

    async def mock_load_completion(self, *args, **kwargs):
        """Mock completion for load testing"""
        await asyncio.sleep(0.001)  # Minimal delay to simulate network
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Use gpt-4"
        response.usage = {"total_tokens": 50}
        return response

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_sustained_load(self, mock_acompletion, load_test_models):
        """Test sustained load over time"""
        mock_acompletion.side_effect = self.mock_load_completion

        selector = LLMSelector(load_test_models)

        # Run sustained load for 5 seconds
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < 5.0:
            result = await selector.select(f"load test {request_count}")
            assert result == "gpt-4"
            request_count += 1

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate throughput
        throughput = request_count / total_time  # requests per second

        # Should handle reasonable throughput
        assert throughput > 10  # At least 10 requests per second
        assert request_count > 50  # At least 50 requests in 5 seconds

        metrics = selector.get_metrics()
        assert metrics["total_requests"] == request_count
        assert metrics["successful_selections"] == request_count

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_burst_load(self, mock_acompletion, load_test_models):
        """Test burst load handling"""
        mock_acompletion.side_effect = self.mock_load_completion

        selector = LLMSelector(load_test_models)

        # Burst of 50 concurrent requests
        burst_size = 50
        tasks = [selector.select(f"burst test {i}") for i in range(burst_size)]

        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        total_time = end_time - start_time

        # All should succeed
        assert len(results) == burst_size
        assert all(r == "gpt-4" for r in results)

        # Calculate metrics
        avg_latency = total_time / burst_size
        throughput = burst_size / total_time

        # Burst should complete reasonably quickly
        assert total_time < 2.0  # Less than 2 seconds for 50 requests
        assert avg_latency < 0.1  # Less than 100ms per request
        assert throughput > 25  # At least 25 requests per second

    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_memory_stability_under_load(self, mock_acompletion, load_test_models):
        """Test memory stability under sustained load"""
        mock_acompletion.side_effect = self.mock_load_completion

        selector = LLMSelector(load_test_models, cache_enabled=True, max_cache_size=100)

        # Run many requests
        num_requests = 500

        for i in range(num_requests):
            result = await selector.select(f"memory stability test {i}")
            assert result == "gpt-4"

            # Periodic cache size check
            if i % 100 == 0:
                cache_size = len(selector._cache)
                assert cache_size <= 100

        # Final cache size check
        final_cache_size = len(selector._cache)
        assert final_cache_size <= 100

        # Metrics should be accurate
        metrics = selector.get_metrics()
        assert metrics["total_requests"] == num_requests
        assert metrics["successful_selections"] == num_requests


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance", "--durations=10"])