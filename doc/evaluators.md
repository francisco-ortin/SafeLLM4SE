# Evaluator Guide

`safellm4se-sample` is evaluator-agnostic, so it can be used with any user-provided evaluator, as long as it implements the `Evaluator` contract.
An evaluator is a Python class that
implements the `safellm4se.sampling.evaluators.Evaluator` contract and returns
one numeric observation at a time.

We provide example evaluators for local random sampling and the [Ollama](https://www.ollama.com), [Gemini](https://gemini.google.com), and [Groq](https://groq.com/) APIs.
For these API providers, we provide default evaluators to help you get started with the evaluation of
LLMs served by these platforms. You can also implement your own evaluator for any other LLM provider or evaluation process.

## Loading evaluators

The `--evaluator` argument accepts:

- a module containing exactly one concrete evaluator class; or
- an explicit `module:ClassName` reference.

Examples:

```bash
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.random_binary_evaluator
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.random_binary_evaluator:RandomBinaryEvaluator
```

## Evaluator contract

An evaluator must expose:

| Member | Meaning |
|---|---|
| `run(**context)` | Executes one observation. |
| `theta` | Numeric result for the last run. |
| `metric_type` | `binary` or `continuous`. |
| `experiment_name` | Name of the experiment. |
| `model_name` | Human-readable model label. |
| `model_id` | Provider-specific model identifier. |
| `prompt_tokens` | Prompt tokens used by the last run. |
| `completion_tokens` | Completion tokens used by the last run. |
| `total_tokens` | Total tokens used by the last run. |

`run` may return:

- `SamplingObservation`;
- `bool`, converted to `0.0` or `1.0`;
- `int` or `float`, converted to a continuous numeric observation;
- a dictionary containing `theta` or `quality`, plus optional token fields.

If `run` returns `None`, the sampler reads the public properties from the
evaluator instance.

## Minimal custom evaluator

```python
"""Example evaluator used by SafeLLM4SE documentation."""

from typing import Any

from safellm4se.sampling.evaluators import Evaluator


class ConstantEvaluator(Evaluator):
    """Evaluator that always reports the same continuous value."""

    def __init__(self, value: float = 0.5, **parameters: Any) -> None:
        """Initialize the evaluator.

        Args:
            value: Numeric theta returned by each run.
            **parameters: Extra evaluator parameters accepted for compatibility.
        """
        super().__init__(**parameters)
        self._value: float = float(value)  # Numeric theta returned by each run.

    def run(self, **context: Any) -> float:
        """Return one constant observation.

        Args:
            **context: Runtime values supplied by the sampler.

        Returns:
            The configured constant theta value.
        """
        del context
        return self._value

    @property
    def theta(self) -> float:
        """Return the last theta value.

        Returns:
            The constant theta value.
        """
        return self._value

    @property
    def metric_type(self) -> str:
        """Return the metric type.

        Returns:
            The continuous metric type.
        """
        return "continuous"

    @property
    def experiment_name(self) -> str:
        """Return the experiment name.

        Returns:
            The evaluator experiment name.
        """
        return "constant"

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            The evaluator model name.
        """
        return "constant"

    @property
    def model_id(self) -> str:
        """Return the model identifier.

        Returns:
            The evaluator model identifier.
        """
        return "constant-v0"

    @property
    def prompt_tokens(self) -> int:
        """Return prompt token usage.

        Returns:
            Zero prompt tokens.
        """
        return 0

    @property
    def completion_tokens(self) -> int:
        """Return completion token usage.

        Returns:
            Zero completion tokens.
        """
        return 0
```

Run it with:

```bash
safellm4se-sample --evaluator path.to.module:ConstantEvaluator -- value=0.75
```

## Included evaluators

### Local random evaluators

| Module | Class | Metric | Main parameters |
|---|---|---|---|
| `safellm4se.sampling.myevaluators.random_binary_evaluator` | `RandomBinaryEvaluator` | `binary` | `success_probability`, `model_id` |
| `safellm4se.sampling.myevaluators.random_normal_evaluator` | `RandomNormalEvaluator` | `continuous` | `mean`, `standard_deviation`, `model_id` |

### [Ollama](https://ollama.com/) evaluators

| Module | Class | Metric | Main parameters |
|---|---|---|---|
| `safellm4se.sampling.myevaluators.ollama.random` | `OllamaRandomEvaluator` | `continuous` | `prompt`, `temperature`, `model_id`, `model_name`, `max_tokens`, `ollama_host`, `system_prompt` |
| `safellm4se.sampling.myevaluators.ollama.humaneval_oneprogram` | `OllamaHumanEvalOneProgramEvaluator` | `binary` | `problem_number`, `test_timeout`, `temperature`, `model_id`, `model_name`, `max_tokens`, `ollama_host`, `system_prompt` |
| `safellm4se.sampling.myevaluators.ollama.humaneval_fullbench` | `OllamaHumanEvalFullBenchEvaluator` | `continuous` | `test_timeout`, `temperature`, `max_tokens`, `ollama_host`, `system_prompt` |

Notes:

- [Ollama](https://ollama.com/) requests are sent to `/api/chat`.
- The default host is `http://host.docker.internal:11434`.
- `humaneval_fullbench` currently fixes `model_name` to `deepseek-coder` and
  `model_id` to `deepseek-coder:6.7b` through properties in that module.

### [Gemini](https://gemini.google.com/) evaluators

| Module | Class | Metric | Main parameters |
|---|---|---|---|
| `safellm4se.sampling.myevaluators.gemini.random` | `GeminiRandomEvaluator` | `continuous` | `prompt`, `temperature`, `model_id`, `model_name`, `max_tokens`, `api_keys_file`, `system_prompt` |
| `safellm4se.sampling.myevaluators.gemini.humaneval_oneprogram` | `GeminiHumanEvalOneProgramEvaluator` | `binary` | `problem_number`, `test_timeout`, `temperature`, `model_id`, `model_name`, `max_tokens`, `api_keys_file`, `system_prompt` |
| `safellm4se.sampling.myevaluators.gemini.humaneval_fullbench` | `GeminiHumanEvalFullBenchEvaluator` | `continuous` | `test_timeout`, `temperature`, `model_id`, `model_name`, `max_tokens`, `api_keys_file`, `system_prompt` |

Default model identifier: `gemini-3.1-flash-lite`.

### [Groq](https://groq.com/) evaluators

| Module | Class | Metric | Main parameters |
|---|---|---|---|
| `safellm4se.sampling.myevaluators.groq.random` | `GroqRandomEvaluator` | `continuous` | `prompt`, `temperature`, `model_id`, `model_name`, `max_tokens`, `api_keys_file`, `api_key_name`, `system_prompt` |
| `safellm4se.sampling.myevaluators.groq.humaneval_oneprogram` | `GroqHumanEvalOneProgramEvaluator` | `binary` | `problem_number`, `test_timeout`, `temperature`, `model_id`, `model_name`, `max_tokens`, `api_keys_file`, `api_key_name`, `system_prompt` |
| `safellm4se.sampling.myevaluators.groq.humaneval_fullbench` | `GroqHumanEvalFullBenchEvaluator` | `continuous` | `test_timeout`, `temperature`, `model_id`, `model_name`, `max_tokens`, `api_keys_file`, `api_key_name`, `system_prompt` |

Default model identifier: `openai/gpt-oss-20b`.

## HumanEval evaluation

The [HumanEval](https://humaneval.org/) evaluators load `openai/openai_humaneval` through the `datasets`
package. They ask a model to complete Python functions and execute the generated
code against the benchmark tests in a separate process with a timeout.

[HumanEval](https://humaneval.org/) modes:

- `humaneval_oneprogram`: evaluates one HumanEval problem and returns a binary
  pass/fail `theta`.
- `humaneval_fullbench`: evaluates all 164 HumanEval problems and returns the
  pass rate as continuous `theta`.

Generated code is executed locally. Run these evaluators only in an environment
where executing model-generated Python code is acceptable.
