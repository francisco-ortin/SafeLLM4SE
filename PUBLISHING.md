# Publishing SafeLLM4SE

This project is configured for PyPI publication as `safellm4se`.

## Build

Use a clean virtual environment with Python 3.10 or newer:

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
```

## TestPyPI

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Install the uploaded package from TestPyPI:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ safellm4se
```

## PyPI

After validating the TestPyPI package, upload the same built artifacts to PyPI:

```bash
python -m twine upload dist/*
```

Before each release, update the version in `pyproject.toml` and
`src/safellm4se/__init__.py`.
