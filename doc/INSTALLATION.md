# Installation

This document lists the minimum runtime packages and the optional packages used
by the example evaluators in `sampling/myevaluators`.

## Python Version

Use Python 3.10 or newer. The code uses modern type-hint syntax such as
`Path | None`.

## Minimal Installation

The framework core uses the Python standard library for CSV persistence,
adaptive sampling, confidence intervals, and file locking. The only required
third-party runtime dependency for the command-line programs is:

- `loguru`: execution logging.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install loguru
```

With this minimal installation you can run:

- `sample.py` with custom evaluators that do not need extra libraries.
- `sample.py` with the random example evaluators.
- `report.py` and `compare.py` without plots.

## Recommended Statistical Extras

SafeLLM4SE works without SciPy, but `scipy` improves several calculations:

- Student t critical values.
- Shapiro-Wilk and Anderson-Darling normality checks.
- Mann-Whitney U and Wilcoxon exact/library implementations.
- Gaussian KDE when generating KDE-based plots.

```bash
python -m pip install scipy
```

When SciPy is not available, SafeLLM4SE uses standard-library fallbacks for the
normal quantile, t-like intervals, non-parametric test approximations, and KDE.

## Plotting

Install Matplotlib only if you want SVG plots from `report.py` or `compare.py`:

```bash
python -m pip install matplotlib
```

Plot options that need Matplotlib:

- `--boxplot`
- `--violin`
- `--ecdf`
- `--raincloud`
- `--kde`

## Example Evaluator Dependencies

The directory `sampling/myevaluators` contains optional example evaluators.
Install their dependencies only when using them.

| Evaluator family | Purpose | Packages |
|---|---|---|
| Random examples | Local random binary or continuous observations | No extra package |
| Ollama random | Calls a local Ollama model | No Python package; requires Ollama service |
| Ollama HumanEval | Calls Ollama and loads HumanEval | `datasets` |
| Gemini random | Calls Gemini API | `google-genai` |
| Gemini HumanEval | Calls Gemini API and loads HumanEval | `google-genai datasets` |
| Groq random | Calls Groq API | `groq` |
| Groq HumanEval | Calls Groq API and loads HumanEval | `groq datasets` |

Install all optional packages used by the included examples:

```bash
python -m pip install scipy matplotlib datasets google-genai groq
```

The Dockerfile currently installs `pandas`, `statsmodels`, `groq`,
`google-genai`, `datasets`, `loguru`, `pyarrow`, and `matplotlib`, but `pandas`,
`statsmodels`, and `pyarrow` are not required by the current core Python code.

## Provider Setup

### Ollama

Ollama evaluators call `/api/chat` through Python's standard HTTP library. Start
Ollama and pull the model used by the evaluator:

```bash
ollama pull qwen2.5-coder:7b
ollama pull deepseek-coder:6.7b
```

The default host in the code is `http://host.docker.internal:11434`, which is
convenient from Docker. When running directly on the host, pass:

```bash
-- ollama_host=http://localhost:11434
```

### Gemini

Install the SDK:

```bash
python -m pip install google-genai
```

Create `sampling/myevaluators/api-keys.json` from the example file and set the
`gemini` key:

```json
{
  "gemini": "your Gemini API key"
}
```

### Groq

Install the SDK:

```bash
python -m pip install groq
```

Create `sampling/myevaluators/api-keys.json` from the example file and set the
`groq` key:

```json
{
  "groq": "your Groq API key"
}
```

You can override the path with an evaluator parameter:

```bash
python sample.py --evaluator sampling.myevaluators.groq.random -- api_keys_file="secrets/api-keys.json"
```
