# Contributing to ComfyUI-Wan22Blockswap

Thank you for your interest in contributing to ComfyUI-Wan22Blockswap! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

-   [Code of Conduct](#code-of-conduct)
-   [How to Contribute](#how-to-contribute)
-   [Development Setup](#development-setup)
-   [Coding Standards](#coding-standards)
-   [Pull Request Process](#pull-request-process)
-   [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We're all here to make this project better together.

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

-   **Bug Reports**: Found a bug? Let us know!
-   **Feature Requests**: Have an idea for improvement? Share it!
-   **Code Contributions**: Bug fixes, new features, optimizations
-   **Documentation**: Improve README, add examples, fix typos
-   **Testing**: Help test on different hardware configurations

### Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your contribution
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Prerequisites

-   Python 3.8 or higher
-   ComfyUI installed and working
-   Git

### Local Development

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ComfyUI_Wan22Blockswap.git

# Navigate to the project
cd ComfyUI_Wan22Blockswap

# Create a development branch
git checkout -b feature/your-feature-name
```

### Running Tests

```bash
# Navigate to test_scripts directory
cd test_scripts

# Run unit tests
python test_blockswap_looper.py

# Run integration tests
python test_looper_integration.py

# Validate implementation structure
python validate_implementation.py

# Run performance tests (requires GPU)
python test_performance.py

# Run real-world performance tests (requires GPU and models)
python test_real_performance.py
```

## Coding Standards

### Python Style

We follow PEP 8 with the following specifics:

-   **Indentation**: 4 spaces (no tabs)
-   **Line length**: Maximum 79 characters
-   **Quotes**: Use double quotes for strings
-   **Type hints**: Required for all function parameters and returns

### Documentation

-   All functions must have docstrings following PEP 257
-   Include parameter descriptions and return value documentation
-   Add inline comments for complex logic

### Example Function

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.

    Parameters:
        param1 (str): Description of param1.
        param2 (int): Description of param2.

    Returns:
        bool: Description of return value.

    Raises:
        ValueError: When param2 is negative.
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
    return True
```

### Logging

-   Use the `[BlockSwap]` prefix for debug messages
-   Log important state changes and errors
-   Avoid excessive logging in production code

```python
if debug:
    print(f"[BlockSwap] Processing block {block_index}")
```

### Error Handling

-   Use specific exception types
-   Provide helpful error messages
-   Clean up resources in finally blocks when appropriate

## Pull Request Process

### Before Submitting

1. **Test your changes**: Run all tests and ensure they pass
2. **Check code style**: Run linting tools (black, flake8)
3. **Update documentation**: If needed, update README or docstrings
4. **Write descriptive commits**: Use clear commit messages

### PR Guidelines

-   **Title**: Brief, descriptive summary of changes
-   **Description**: Explain what, why, and how
-   **Link issues**: Reference any related issues
-   **Small PRs**: Prefer smaller, focused pull requests

### PR Template

```markdown
## Description

Brief description of changes made.

## Type of Change

-   [ ] Bug fix
-   [ ] New feature
-   [ ] Documentation update
-   [ ] Performance improvement
-   [ ] Code refactoring

## Testing

Describe how you tested your changes.

## Checklist

-   [ ] Code follows project style guidelines
-   [ ] Self-review completed
-   [ ] Documentation updated if needed
-   [ ] Tests added/updated
-   [ ] All tests pass
```

### Review Process

1. Submit your PR
2. Maintainers will review within a few days
3. Address any feedback
4. Once approved, your PR will be merged

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

-   **ComfyUI version**: Your ComfyUI version
-   **Python version**: Output of `python --version`
-   **GPU/VRAM**: Your GPU model and VRAM amount
-   **Model used**: Which WAN model you're using
-   **Steps to reproduce**: Detailed steps to trigger the bug
-   **Expected behavior**: What should happen
-   **Actual behavior**: What actually happens
-   **Error logs**: Any error messages or stack traces

### Feature Requests

When requesting features, please include:

-   **Use case**: Why you need this feature
-   **Proposed solution**: How you envision it working
-   **Alternatives**: Other solutions you've considered

## Questions?

If you have questions about contributing, feel free to:

-   Open a discussion on GitHub
-   Create an issue with the "question" label

Thank you for contributing! 🎉
