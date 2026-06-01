# Contributing

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dev dependencies:
   ```bash
   pip install -e ".[dev,ui]"
   ```

## Running Tests

```bash
pytest
```

All tests must pass before submitting a pull request.

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings to public functions and classes
- Keep modules focused — one responsibility per file

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commits
3. Ensure all tests pass
4. Update documentation if needed
5. Open a pull request with a clear description of the changes

## Adding a New Detector

To add a new quality detector:

1. Create a new file in `analyzer/detectors/`
2. Implement your detector following the existing pattern (see `quality.py` or `pii.py`)
3. Register it in the pipeline (`analyzer/pipeline.py`)
4. Add tests in `tests/`

## Reporting Issues

Please open a GitHub issue with:
- A clear description of the bug or feature request
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Python version and OS
