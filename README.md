# LLM Selector

A Python library that uses any LLM (via LiteLLM) to intelligently select the best model for a given task.

## Installation

### From PyPI (Recommended)

```bash
pip install llm-selector
```

### From Source

```bash
git clone https://github.com/YoannDev90/llm-selector.git
cd llm-selector
pip install .
```

### Development Installation

```bash
git clone https://github.com/YoannDev90/llm-selector.git
cd llm-selector
pip install -e ".[test]"
```

## Quick Start

```python
import asyncio
from llm_selector import LLMSelector, LLMSelectorConfig

async def main():
    # Define available models
    models = {
        "gpt-4": {
            "description": "OpenAI GPT-4 for complex reasoning",
            "weight": 2.0,
            "cost_per_token": 0.00003
        },
        "claude-3": {
            "description": "Anthropic Claude 3 for balanced tasks",
            "weight": 1.0,
            "cost_per_token": 0.000015
        }
    }
    
    # Configure selector LLM (the one that makes the decision)
    selector_configs = [
        LLMSelectorConfig("gpt-3.5-turbo", api_key="your-openai-key")
    ]
    
    # Create selector
    selector = LLMSelector(models=models, selector_configs=selector_configs)
    
    # Select model for a task
    result = await selector.select("Write a complex Python data analysis script")
    print(f"Selected: {result}")

asyncio.run(main())
```

```python
import asyncio
from llm_selector import LLMSelector, LLMSelectorConfig

async def main():
    # Configuration des modèles disponibles avec poids et coûts
    models = {
        "openai/gpt-4": {
            "description": "OpenAI GPT-4 for complex reasoning and coding",
            "weight": 2.0,  # 2x plus de chances
            "cost_per_token": 0.00003  # Coût estimé par token
        },
        "anthropic/claude-3": {
            "description": "Anthropic Claude 3 for safe and helpful responses", 
            "weight": 1.5,
            "cost_per_token": 0.000015
        },
        "cerebras/llama3.3-70b": {
            "description": "Cerebras Llama 3.3 70B for fast inference",
            "weight": 1.0
        },
    }
    
    # Configurations des services de sélection avec fallbacks
    selector_configs = [
        LLMSelectorConfig(
            model="openrouter/openai/gpt-oss-20b",
            api_base="https://openrouter.ai/api/v1",
            api_key="env:OPENROUTER_API_KEY"  # Variable d'environnement
        ),
        LLMSelectorConfig(
            model="anthropic/claude-3-haiku",
            api_key="dotenv:ANTHROPIC_KEY"  # Clé dans .env
        ),
    ]
    
    selector = LLMSelector(
        models=models, 
        selector_configs=selector_configs,
        cache_enabled=True,
        debug=True
    )
    
    user_input = "I need to write a complex Python script for data analysis."
    selected_model = await selector.select(user_input)
    print(f"Selected model: {selected_model}")
    
    # Métriques
    metrics = selector.get_metrics()
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    print(f"Total cost: ${selector.get_total_cost():.4f}")

asyncio.run(main())
```

## Advanced Features

### Synchronous Interface

```python
selector = LLMSelector(models=models, selector_configs=configs)
result = selector.select_sync("My query")  # Synchronous version
```

### Caching

```python
selector = LLMSelector(
    models=models,
    cache_enabled=True,      # Enable caching
    cache_ttl=3600,          # 1 hour TTL
    max_cache_size=1000      # Limit cache size
)

# Clear cache when needed
selector.clear_cache()
```

### Cost Tracking

```python
# Set budget limit
selector.set_budget_limit(10.0)  # $10 limit

# Check if exceeded
if selector.is_budget_exceeded():
    print("Budget exceeded!")

# Get cost breakdown
total_cost = selector.get_total_cost()
cost_by_model = selector.get_metrics()['cost_by_model']
```

### Metrics & Monitoring

```python
metrics = selector.get_metrics()
print(f"""
Requests: {metrics['total_requests']}
Cache hits: {metrics['cache_hits']} ({metrics['cache_hit_rate']:.1%})
Success rate: {metrics['successful_selections']/metrics['total_requests']:.1%}
Average latency: {metrics['average_latency']:.2f}s
Total cost: ${metrics['total_cost']:.4f}
""")

# Reset metrics
selector.reset_metrics()
```

### Cost Tracking

```python
# Set budget limit
selector.set_budget_limit(10.0)  # $10 limit

# Check if exceeded
if selector.is_budget_exceeded():
    print("Budget exceeded!")

# Get cost breakdown
total_cost = selector.get_total_cost()
cost_by_model = selector.get_metrics()['cost_by_model']
```

### Retry Logic

```python
selector = LLMSelector(
    models=models,
    max_retries=3,      # Max retry attempts
    retry_delay=1.0     # Base delay in seconds
)
# Automatic exponential backoff for network/rate limit errors
```

## API Reference

### LLMSelector

#### Constructor

```python
LLMSelector(
    models: dict,                           # Required: Model configurations
    preprompt: str = None,                  # System prompt for selector
    selector_configs: list = None,          # LLM configs for selection
    default_model: str = None,              # Fallback model
    validate_configs: bool = True,          # Validate configs on init
    max_retries: int = 3,                   # Max API retry attempts
    retry_delay: float = 1.0,               # Base retry delay (seconds)
    cache_enabled: bool = True,             # Enable result caching
    cache_ttl: int = 3600,                  # Cache TTL (seconds)
    debug: bool = False,                    # Enable debug logging
    max_cache_size: int = 1000,             # Max cache entries
    **litellm_kwargs                         # Additional litellm params
)
```

#### Methods

- `select(input: str) -> str`: Async model selection
- `select_sync(input: str) -> str`: Sync model selection
- `get_metrics() -> dict`: Get performance metrics
- `reset_metrics()`: Reset all metrics
- `clear_cache()`: Clear cached results
- `get_total_cost() -> float`: Get total API cost
- `set_budget_limit(limit: float)`: Set cost budget
- `is_budget_exceeded() -> bool`: Check budget status
- `validate_configuration() -> bool`: Validate current config

### LLMSelectorConfig

#### Constructor

```python
LLMSelectorConfig(
    model: str,                    # Required: Model identifier
    api_base: str = None,          # Custom API base URL
    api_key: str = None,           # API key (plain/env/dotenv)
    **kwargs                       # Additional litellm parameters
)
```

#### Methods

- `validate() -> bool`: Validate configuration

### Model Configuration Format

```python
models = {
    "model_name": {
        "description": "Human-readable description",    # Required
        "weight": 1.0,                                  # Optional, default 1.0
        "cost_per_token": 0.00001                       # Optional, for cost tracking
    }
}
```

## Configuration

### API Key Sources

API keys can be specified in three ways:

- **Plain text**: `"sk-your-api-key"`
- **Environment variable**: `"env:VARIABLE_NAME"`
- **.env file**: `"dotenv:KEY_NAME"`

### Environment Variables

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-key
```

### Model Weights

Model weights influence selection probability:
- Higher weights (> 1.0) increase selection chance
- Lower weights (< 1.0) decrease selection chance
- Weights are relative to other models

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD ["python", "your_app.py"]
```

### Production Considerations

- Set appropriate `max_cache_size` based on memory constraints
- Configure `cache_ttl` based on how often model preferences change
- Use environment variables for API keys in production
- Monitor metrics regularly for performance optimization
- Set budget limits to control costs

## Model Configuration

Each model supports:

- `description`: Human-readable description
- `weight`: Selection weight (default: 1.0)
- `cost_per_token`: Cost estimation (optional)

## API Key Configuration

API keys can be specified in three ways:

- **Plain text**: `"sk-your-api-key"`
- **Environment variable**: `"env:VARIABLE_NAME"`
- **.env file**: `"dotenv:KEY_NAME"`

## LLMSelectorConfig

- `model`: The model name (e.g., "openrouter/openai/gpt-4")
- `api_base`: Custom API base URL (optional)
- `api_key`: API key (plain, env:, or dotenv:)
- `**kwargs`: Additional litellm parameters

## Testing

The library includes comprehensive unit tests, integration tests, and performance benchmarks.

### Install Test Dependencies

```bash
pip install -e ".[test]"
```

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest -m "not integration and not performance"

# Integration tests
pytest -m integration

# Performance tests
pytest -m performance

# With coverage
pytest --cov=llm_selector --cov-report=html
```

### Test Coverage

- **Unit Tests**: Core functionality, error handling, configuration validation
- **Integration Tests**: Multi-provider scenarios, cost optimization, concurrent load
- **Performance Tests**: Latency benchmarks, throughput testing, memory usage
- **Coverage Goal**: 80%+ code coverage with detailed edge case testing

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/YoannDev90/llm-selector.git
cd llm-selector
pip install -e ".[test]"
pre-commit install
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=llm_selector --cov-report=html

# Specific test categories
pytest -m "unit"          # Unit tests only
pytest -m "integration"   # Integration tests
pytest -m "performance"   # Performance tests
```

## Changelog

### v0.1.0 (Current)
- Initial release with core LLM selection functionality
- Support for multiple LLM providers via LiteLLM
- Intelligent caching with TTL and size limits
- Cost tracking and budget management
- Comprehensive metrics and monitoring
- Retry logic with exponential backoff
- Async and sync interfaces
- Extensive test coverage

## Publishing

For information on how to publish new versions to PyPI, see [PUBLISHING.md](PUBLISHING.md).

The project includes automated publishing via GitHub Actions that triggers on version tags.