# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- Core LLM selection functionality
- Multi-provider support via LiteLLM
- Intelligent caching system
- Cost tracking and budget management
- Comprehensive metrics and monitoring
- Retry logic with exponential backoff
- Async and sync interfaces
- Extensive test suite with 80%+ coverage

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2024-01-XX

### Added
- Complete LLMSelector implementation with all planned features
- LLMSelectorConfig for provider configuration
- Support for multiple API key sources (plain, env, dotenv)
- Model weight-based selection preferences
- Cache with TTL and size limits
- Cost estimation and tracking
- Budget limit functionality
- Performance metrics collection
- Error handling and retry mechanisms
- Comprehensive test coverage
- Documentation and examples

### Technical Details
- Python 3.8+ compatibility
- Async/await support
- Type hints throughout
- Extensive error handling
- Memory-efficient caching
- Configurable retry policies