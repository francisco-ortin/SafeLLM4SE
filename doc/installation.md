# Installation

This document lists the minimum runtime packages and the optional packages used
by SafeLLM4SE and the example evaluators in `safellm4se.sampling.myevaluators`.

## Python version

Use Python 3.10 or newer (the code uses modern type-hint syntax such as
`Path | None`).

## Easiest installation

The easiest way to install SafeLLM4SE is to install the package from PyPI:

```bash
pip install "SafeLLM4SE[all]"
```

## Installation from source code

To install SafeLLM4SE from its source code, clone the repository and run:

```bash
git clone https://github.com/francisco-ortin/SafeLLM4SE.git
cd SafeLLM4SE
```

The framework core uses the Python standard library for CSV persistence,
adaptive sampling, confidence intervals, and file locking.
It also uses the following packages:
- `loguru` for logging.
- `scipy` for statistical calculations.
- `matplotlib` if you want to generate plots.

```bash
pip install loguru scipy matplotlib
```

With this minimal installation you can run:

- `safellm4se-sample` with custom evaluators that do not need extra libraries.
- `safellm4se-sample` with the random example evaluators.
- `safellm4se-report` and `safellm4se-compare`.


### [Ollama](https://ollama.com/)

SafeLLM4SE can call [Ollama](https://ollama.com/) models through the local Ollama service.
You can use the `BaseEvaluator` class in `safellm4se.sampling.myevaluators.ollama` to implement your own evaluator, or you can use the example evaluators provided in the same module.
These services do not require any extra Python package, but you need to install the [Ollama](https://ollama.com/)
service and pull the models you want to use.

Once installed, pull the models used by the evaluator:

```bash
ollama pull qwen2.5-coder:7b
ollama pull deepseek-coder:6.7b
```

The default host in the code is `http://host.docker.internal:11434`, which is
convenient from Docker, but you can change it to `http://localhost:11434` if you run the code outside Docker.

### [Gemini](https://gemini.google.com/)

You have a `BaseEvaluator` class in `safellm4se.sampling.myevaluators.gemini` to implement your own evaluator,
or you can use the example evaluators provided in the same module.
The [Gemini](https://gemini.google.com/) evaluators use the `google-genai` package to call the Gemini API.

```bash
pip install google-genai
```

Obtain a [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) from [Google Cloud](https://console.cloud.google.com/apis/credentials) and set the `gemini` key in `api-keys.json` in your working directory:

```json
{
  "gemini": "your Gemini API key"
}
```

Then, pass the path to `api-keys.json` to the evaluator with the `--api_keys_file` parameter.

### [Groq](https://groq.com/)

You have a `BaseEvaluator` class in `safellm4se.sampling.myevaluators.groq` to implement your own evaluator,
or you can use the example evaluators provided in the same module.
The [Groq](https://groq.com/) evaluators use the `groq` package to call the [Groq](https://groq.com/) API.

```bash
pip install groq
```

Obtain a [Groq](https://groq.com/) API key from [Groq](https://console.groq.com/) and set the `groq` key in `api-keys.json` in your working directory:
```json
{
  "groq": "your Groq API key"
}
```

Then, pass the path to `api-keys.json` to the evaluator with the `--api_keys_file` parameter.

### [HumanEval](https://humaneval.org/)

[HumanEval](https://humaneval.org/) is a benchmark for evaluating code generation models. 
We prodive example evaluators that use HumanEval in the `safellm4se.sampling.myevaluators` package.
To run those example evaluators, you must install the `datasets` package to load the [HumanEval](https://humaneval.org/) benchmark.

```bash
pip install datasets
```


## Example evaluator dependencies

The package module `safellm4se.sampling.myevaluators` contains optional example
evaluators. Install their dependencies only when using them.

| Evaluator family | Purpose | Packages |
|---|---|---|
| Random examples | Local random binary or continuous observations | No extra package |
| Ollama random | Calls a local Ollama model | No Python package; requires Ollama service |
| Ollama HumanEval | Calls Ollama and loads HumanEval | `datasets` |
| Gemini random | Calls Gemini API | `google-genai` |
| Gemini HumanEval | Calls Gemini API and loads HumanEval | `google-genai datasets` |
| Groq random | Calls Groq API | `groq` |
| Groq HumanEval | Calls Groq API and loads HumanEval | `groq datasets` |


## Dockerfile

We provide the Dockerfile used in the development of SafeLLM4SE.
