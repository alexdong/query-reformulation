# Query Reformulation Project Guidelines

## Commands
- Train: `python train/main.py --model-size [small|base|large]`
- Lint: `ruff .`
- Test: `python benchmark/score.py` (for specific test)
- Benchmark: `python benchmark/main.py`

## Code Style
- Python 3.12 with type annotations
- Fail early using assertions, avoid try/except
- Maximize readability, place top-level functions first
- Use print with format `[LEVEL] message` for logging
- One-liners (lambda, list comprehension) preferred
- One empty line between code blocks, two between functions
- Follow ruff configuration in pyproject.toml
- Variable naming: clear intent without type information
- Use /tmp for temporary files, don't clean up
- Include `if __name__ == "__main__"` block in every script
- Prefer simplicity and minimal cognitive load
- Function-based approach over class-based (avoid classes unless necessary)
- Early failure with assertions rather than exception handling