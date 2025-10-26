# Contributing to LLM Selector

Thank you for your interest in contributing to LLM Selector! This document provides guidelines and information for contributors.

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors

## How to Contribute

### 1. Fork and Clone

```bash
git clone https://github.com/YoannDev90/llm-selector.git
cd llm-selector
git checkout -b feature/your-feature-name
```

### 2. Development Setup

```bash
# Install in development mode with test dependencies
pip install -e ".[test]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests to ensure everything works
pytest
```

### 3. Make Changes

- Follow the existing code style (black, isort, flake8)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit and Push

```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

Open a pull request on GitHub with a clear description of your changes.

## Development Guidelines

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks with:
```bash
pre-commit run --all-files
```

### Testing

- Write tests for all new functionality
- Maintain >80% code coverage
- Test both success and failure scenarios
- Include integration tests for complex features

### Documentation

- Update docstrings for public APIs
- Add examples for new features
- Update README.md if needed
- Keep changelog up to date

## Types of Contributions

### Bug Fixes
- Fix bugs in existing code
- Add regression tests
- Update documentation

### Features
- Implement new functionality
- Add comprehensive tests
- Update documentation and examples

### Documentation
- Improve existing documentation
- Add tutorials or guides
- Translate documentation

### Testing
- Add missing test cases
- Improve test coverage
- Fix flaky tests

## Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag
4. Push to main branch
5. GitHub Actions will handle PyPI release

## Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Discord**: For real-time chat (link coming soon)

Thank you for contributing to LLM Selector! 🚀