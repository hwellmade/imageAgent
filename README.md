# Image AI Agent

An intelligent AI agent for image processing and analysis.

## Setup

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Create and activate the virtual environment:
```bash
poetry install
poetry shell
```

3. Run the development server:
```bash
# Command will be added once the main application is developed
```

## Project Structure

```
image_agent/           # Main package directory
├── __init__.py       # Package initialization
├── core/             # Core functionality
├── models/           # ML model implementations
├── api/              # API endpoints
└── utils/            # Utility functions

tests/                # Test directory
```

## Development

- Use `poetry add <package>` to add new dependencies
- Use `poetry add -D <package>` to add development dependencies
- Run tests with `pytest`
- Format code with `ruff format`
- Check types with `mypy`

## License

[MIT License](LICENSE) 