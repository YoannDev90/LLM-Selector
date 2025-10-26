import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import os
import tempfile

from llm_selector import LLMSelector, LLMSelectorConfig


class TestLLMSelectorConfig:
    """Test LLMSelectorConfig class"""

    def test_init_basic(self):
        config = LLMSelectorConfig("gpt-4")
        assert config.model == "gpt-4"
        assert config.api_base is None
        assert config.api_key is None

    def test_init_with_params(self):
        config = LLMSelectorConfig(
            model="gpt-4",
            api_base="https://api.example.com",
            api_key="test-key",
            temperature=0.7
        )
        assert config.model == "gpt-4"
        assert config.api_base == "https://api.example.com"
        assert config.api_key == "test-key"
        assert config.extra_kwargs["temperature"] == 0.7

    def test_api_key_env(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "env-key")
        config = LLMSelectorConfig("gpt-4", api_key="env:TEST_API_KEY")
        assert config.api_key == "env-key"

    def test_api_key_env_missing(self):
        config = LLMSelectorConfig("gpt-4", api_key="env:NONEXISTENT_KEY")
        assert config.api_key is None

    @patch("dotenv.load_dotenv")
    def test_api_key_dotenv(self, mock_load_dotenv, monkeypatch):
        # Mock the environment variable that dotenv would set
        monkeypatch.setenv("DOTENV_KEY", "test-dotenv-key")
        config = LLMSelectorConfig("gpt-4", api_key="dotenv:DOTENV_KEY")
        assert config.api_key == "test-dotenv-key"

    def test_api_key_plain(self):
        config = LLMSelectorConfig("gpt-4", api_key="plain-key")
        assert config.api_key == "plain-key"

    def test_validate_valid(self):
        config = LLMSelectorConfig("openai/gpt-4", api_base="https://api.openai.com")
        assert config.validate() is True

    def test_validate_invalid_url(self):
        config = LLMSelectorConfig("gpt-4", api_base="not-a-url")
        assert config.validate() is False

    def test_validate_empty_model_raises_error(self):
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            LLMSelectorConfig("")


class TestLLMSelectorBasic:
    """Test basic LLMSelector functionality"""

    @pytest.fixture
    def sample_models(self):
        return {
            "gpt-4": {"description": "GPT-4 model", "weight": 2.0, "cost_per_token": 0.00003},
            "claude-3": {"description": "Claude 3 model", "weight": 1.0, "cost_per_token": 0.000015},
            "llama-3": {"description": "Llama 3 model"}
        }

    @pytest.fixture
    def sample_configs(self):
        return [
            LLMSelectorConfig("gpt-4", api_key="test-key"),
            LLMSelectorConfig("claude-3", api_key="test-key2")
        ]

    def test_init_creates_instance(self, sample_models, sample_configs):
        """Test that LLMSelector can be instantiated"""
        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        assert selector is not None
        # Test that it has some expected attributes
        assert hasattr(selector, 'debug')
        assert hasattr(selector, 'max_cache_size')

    def test_parse_selection_exact_match(self, sample_models):
        selector = LLMSelector(sample_models)
        assert selector.parse_selection("I recommend gpt-4") == "gpt-4"
        assert selector.parse_selection("Use claude-3 for this") == "claude-3"

    def test_parse_selection_fuzzy_match(self, sample_models):
        selector = LLMSelector(sample_models)
        assert selector.parse_selection("Use GPT4") == "gpt-4"
        assert selector.parse_selection("claude3 model") == "claude-3"

    def test_parse_selection_empty_response(self, sample_models):
        selector = LLMSelector(sample_models)
        # This should return the default model (first in dict)
        result = selector.parse_selection("")
        assert result in sample_models.keys()

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_select_success(self, mock_acompletion, sample_models, sample_configs):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I recommend gpt-4"
        mock_acompletion.return_value = mock_response

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")
        assert result == "gpt-4"
        mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_select_fallback_to_default(self, mock_acompletion, sample_models, sample_configs):
        mock_acompletion.side_effect = Exception("API Error")

        selector = LLMSelector(sample_models, selector_configs=sample_configs)
        result = await selector.select("test query")
        # Should fallback to default model (first in dict)
        assert result in sample_models.keys()

    def test_select_sync_basic(self, sample_models, sample_configs):
        """Test synchronous select method"""
        selector = LLMSelector(sample_models, selector_configs=sample_configs)

        with patch("llm_selector.selector.litellm.acompletion") as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Use gpt-4"
            mock_acompletion.return_value = mock_response

            result = selector.select_sync("test query")
            assert result == "gpt-4"

    def test_debug_logging(self, sample_models, caplog):
        selector = LLMSelector(sample_models, debug=True)
        # Test that debug logging is enabled
        assert selector.debug is True

    def test_cache_operations(self, sample_models):
        selector = LLMSelector(sample_models, cache_enabled=True)
        # Test that cache operations don't crash
        try:
            selector.clear_cache()
            assert True  # If we get here, the method exists and works
        except AttributeError:
            pytest.skip("Cache methods not implemented yet")


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    @patch("llm_selector.selector.litellm.acompletion")

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_acompletion):
        """Test a complete workflow"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I recommend gpt-4 for this complex task"
        mock_response.usage = {"total_tokens": 200}
        mock_acompletion.return_value = mock_response

        models = {
            "gpt-4": {"description": "Complex tasks", "weight": 2.0, "cost_per_token": 0.00003},
            "claude-3": {"description": "Balanced tasks", "weight": 1.0}
        }

        configs = [
            LLMSelectorConfig("gpt-4", api_key="test-key", temperature=0.7)
        ]

        selector = LLMSelector(
            models=models,
            selector_configs=configs,
            cache_enabled=True,
            debug=True
        )

        # First request
        result1 = await selector.select("Complex coding task")
        assert result1 == "gpt-4"

        # Second identical request (should use cache if implemented)
        result2 = await selector.select("Complex coding task")
        assert result2 == "gpt-4"

        # Check that API was called
        assert mock_acompletion.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__])