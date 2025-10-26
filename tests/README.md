# LLM Selector Tests

This directory contains comprehensive unit tests, integration tests, and performance benchmarks for the LLM Selector library.

## Test Structure

```
tests/
├── __init__.py                 # Test package marker
├── test_llm_selector.py        # Main unit tests for LLMSelector class
├── test_error_handling.py      # Error handling and edge case tests
├── test_integration.py         # Integration tests for complex scenarios
├── test_performance.py         # Performance benchmarks and load tests
└── README.md                   # This file
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -e ".[test]"
# or
pip install -r requirements-test.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m "not integration and not performance"

# Integration tests only
pytest -m integration

# Performance tests only
pytest -m performance

# Run with coverage
pytest --cov=llm_selector --cov-report=html
```

### Run Individual Test Files

```bash
pytest tests/test_llm_selector.py
pytest tests/test_error_handling.py
pytest tests/test_integration.py -v
pytest tests/test_performance.py --durations=10
```

## Test Categories

### Unit Tests (`test_llm_selector.py`)
- LLMSelectorConfig validation and initialization
- LLMSelector core functionality
- Cache operations
- Metrics tracking
- Cost calculation and budget management
- Parsing and selection logic

### Error Handling Tests (`test_error_handling.py`)
- API failure scenarios and fallbacks
- Invalid response handling
- Configuration validation errors
- Edge cases in parsing
- Network timeout simulation
- Authentication and rate limit errors

### Integration Tests (`test_integration.py`)
- Complex multi-provider scenarios
- Cost optimization with budget constraints
- Provider fallback mechanisms
- Performance monitoring
- Cache performance optimization
- Retry logic with exponential backoff
- Concurrent load testing
- Memory efficiency testing

### Performance Tests (`test_performance.py`)
- Latency benchmarks
- Cache performance improvement measurement
- Concurrent request handling
- Memory usage patterns
- Scaling performance with model count
- Load testing for high-throughput scenarios

## Test Configuration

The test suite is configured via `pytest.ini` with:
- Coverage reporting (80% minimum threshold)
- Verbose output
- Short traceback format
- Async test support
- Custom markers for different test types

## Mocking Strategy

Tests use comprehensive mocking to:
- Avoid real API calls during testing
- Control response timing and content
- Test error scenarios safely
- Enable fast, reliable test execution
- Support concurrent testing

## Performance Benchmarks

Performance tests measure:
- Selection latency (target: <100ms average)
- Cache hit ratio improvement
- Concurrent request throughput
- Memory usage under load
- Scaling characteristics

## Coverage Goals

The test suite aims for:
- **80%+ code coverage** across all modules
- **100% coverage** of critical paths
- Edge case coverage for robustness
- Performance regression detection

## Continuous Integration

Tests are designed to run in CI environments with:
- No external dependencies
- Fast execution (<30 seconds for unit tests)
- Deterministic results
- Clear failure reporting

## Adding New Tests

When adding new tests:

1. **Unit Tests**: Focus on individual methods and classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Measure and benchmark performance
4. **Error Tests**: Cover failure scenarios and edge cases

Use descriptive test names and include docstrings explaining the test purpose.

## Test Data

Tests use realistic but mocked data:
- Model configurations with weights and costs
- API responses with usage statistics
- Error conditions and recovery scenarios
- Performance timing measurements

## Debugging Tests

For debugging failing tests:
```bash
# Run with detailed output
pytest -v -s

# Run specific test with debugging
pytest tests/test_llm_selector.py::TestLLMSelector::test_select_success -v -s

# Run with coverage details
pytest --cov=llm_selector --cov-report=term-missing
```