# AGENTS.md - Codex Project Instructions

## Code Style

- All Python code, variables, functions, and classes must be named in English.
- All inline comments, docstrings, and technical explanations inside code blocks must be strictly in English.
- Follow PEP 8 guidelines.
- Annotate all function parameters, return values, variables, and class attributes with Python type hints.
- Use descriptive names for variables, functions, and classes to improve readability.
- Avoid abbreviations or single-letter variable names unless they are widely accepted in the Python community.
- Refactor code to improve readability, maintainability, and performance when necessary.
- Every function must include a docstring that describes its purpose, parameters, and return value when applicable.
- Every module must include a docstring that explains its responsibility and expected usage.

## Project Notes

- This project evaluates LLM code-generation reliability using the SAFE protocol on HumanEval-style tasks.
- Runtime measurements are written under `results/`.
- Logs are written under `logs/`.
- API keys are read from `api-keys.json`.
- LLM providers are implemented under `llms/`.
- CSV persistence and summary generation are implemented under `csvutils/`.
