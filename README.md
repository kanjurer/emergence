# Emergence

A research environment for studying the development of intelligence and internal representations in neural networks.

## Development Setup

### Prerequisites
- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Git

### Install dependencies

```bash
uv sync
```

### Run the project

```bash
uv run python scripts/playground.py
```

### Development Commands

| Command | Description |
|---|---|
| `uv run python scripts/playground.py` | Run experimental playground |
| `uv run python scripts/check_environment.py` | Check Python and hardware |
| `uv run pytest -v` | Run automated tests |
| `uv run ruff check .` | Check code quality |
| `uv run ruff format .` | Format Python code |
| `uv run jupyter lab` | Launch JupyterLab |

### Dependency Management

```bash
uv add PACKAGE_NAME
uv add --dev PACKAGE_NAME
uv remove PACKAGE_NAME
uv sync
```
