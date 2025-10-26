# Publishing Guide

## Automated Publishing with GitHub Actions

This repository includes automated publishing to PyPI using GitHub Actions.

### Setup

#### 1. PyPI API Token

1. Go to [PyPI](https://pypi.org/) and log in
2. Go to Account Settings → API tokens
3. Create a new API token with scope "Entire account"
4. Copy the token

#### 2. Configure Repository Secrets

In your GitHub repository:

1. Go to Settings → Secrets and variables → Actions
2. Add the following secrets:
   - `PYPI_API_TOKEN`: Your PyPI API token (for production releases)
   - `TEST_PYPI_API_TOKEN`: Your TestPyPI API token (for testing)

#### 3. Create TestPyPI Account (Optional but Recommended)

1. Go to [TestPyPI](https://test.pypi.org/)
2. Create an account
3. Generate an API token
4. Add it as `TEST_PYPI_API_TOKEN`

### Publishing Process

#### Automatic Publishing (Recommended)

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.2.0"
   ```

2. **Update CHANGELOG.md** with the new version

3. **Commit and push** your changes:
   ```bash
   git add .
   git commit -m "Release v0.2.0"
   git push
   ```

4. **Create a Git tag**:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

5. **GitHub Actions will automatically**:
   - Build the package
   - Run tests
   - Publish to PyPI
   - Create a GitHub release

#### Manual Testing

You can also trigger the workflow manually to test on TestPyPI:

1. Go to Actions tab in GitHub
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"
4. This will publish to TestPyPI instead of production PyPI

### Version Management

- Use [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`
- Tag format: `v1.0.0`, `v0.1.0`, etc.
- Update version in `pyproject.toml` before tagging
- Update `CHANGELOG.md` with release notes

### Troubleshooting

#### Build Fails
- Check that all dependencies are correctly specified
- Ensure tests pass locally: `pytest`
- Verify the package builds: `python -m build`

#### Publishing Fails
- Check that the API token is correct and has proper permissions
- Ensure the package name is unique on PyPI
- Verify that the version hasn't been published before

#### Workflow Doesn't Trigger
- Make sure the tag follows the pattern `v*.*.*`
- Check that the workflow file is in `.github/workflows/publish.yml`
- Verify repository secrets are set correctly

### Manual Publishing (Fallback)

If automated publishing fails, you can publish manually:

```bash
# Install tools
pip install build twine

# Build
python -m build

# Test on TestPyPI
twine upload --repository testpypi dist/*

# Publish to PyPI
twine upload dist/*
```

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run tests locally: `pytest`
- [ ] Build package locally: `python -m build`
- [ ] Commit changes
- [ ] Create and push git tag
- [ ] Verify GitHub Actions workflow completes successfully
- [ ] Check that package is available on PyPI
- [ ] Verify installation works: `pip install llm-selector`