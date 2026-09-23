# Usage Guide

SafeLLM4SE has a three-step workflow:

1. Collect repeated observations with `safellm4se-sample`.
2. Summarize one task with `safellm4se-report`.
3. Compare two tasks with `safellm4se-compare`.

The central measurement is $\theta$ (`theta`), a numeric property of the evaluation
outcome. For binary tasks it is usually `0.0` or `1.0`. For continuous tasks it
can be a score, pass rate, cost-adjusted quality, or any metric defined by the
evaluator.

## 1. Adaptive sampling

`safellm4se-sample` loads an evaluator class and repeatedly calls it until both
conditions are met:

- at least `--n-min` observations have been collected; and
- either the confidence interval width is at most `--target-ci-width`, or the
  token budget has been reached.

Example with a local random binary evaluator:

```bash
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.random_binary_evaluator --task-id random-binary-demo --n-min 10 --target-ci-width 0.20 --budget-tokens 10000 -- success_probability=0.7
```

Typical console output:

```text
Sample written to output\measurements.csv.
```

The sampler appends rows to `output/measurements.csv` (no existing rows are overwritten). 
Example row produced by measuring the performance of the `qwen2.5-coder:7b` model on the [HumanEval](https://humaneval.org/) benchmark through Ollama:

| date | time | task_id | experiment_name | model_name | model_id | execution_number | prompt_tokens | completion_tokens | total_tokens | theta | metric_type | evaluator | evaluator_parameters | run_id | temperature |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-09-02 | 11:15:35 | task-id-54 | ollama-humaneval-fullbench | qwen-coder | qwen2.5-coder:7b | 1 | 28,081 | 11,703 | 39,784 | 0.8476 | continuous | OllamaHumanEvalFullBenchEvaluator | `{"temperature": 2.0}` | 16835a42-013b-44fc-ba73-002f9b0cb0f4 | 2.0 |

### Passing evaluator parameters

Arguments not recognized by the sampler are passed to the evaluator constructor.
They can be written after the CLI separator `--`, and both forms are accepted:

```bash
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.random_normal_evaluator -- mean=60 standard_deviation=10
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.random_normal_evaluator -- --mean=40 --standard-deviation=20
```

Values are converted with Python literal syntax when possible, so booleans,
numbers, `None`, lists, and dictionaries can be passed directly.

The values of these *free* parameters are ignored by SafeLLM4SE, but the evaluator can use them to configure its behavior.
Their values are passed to the evaluator constructor so the evaluator can use them.
In this way, the evaluator can be configured to run different experiments with the same evaluator class.
One typical use of this feature is to pass the `temperature` parameter to the evaluator, which configures the LLM model temperature.

### Creating a custom evaluator

A user can provide their own evaluator by creating a Python module that defines
an evaluator class derived from `BaseEvaluator`. The module must be importable
from the current Python environment, and the value passed to `--evaluator` must
be the module path without the `.py` extension.

The repository root includes `example_random_evaluator.py`, which defines a
`RandomNormalEvaluator` that returns continuous random scores:

```python
"""Example random continuous normal evaluator for the adaptive sampler."""

import random
from typing import Any

from safellm4se.sampling.myevaluators.base_evaluator import BaseEvaluator
from safellm4se.sampling.models import SamplingObservation

DEFAULT_MEAN: float = 50.0
DEFAULT_STANDARD_DEVIATION: float = 25.0


class RandomNormalEvaluator(BaseEvaluator):
    """Example evaluator that returns real-valued quality scores in [0, 100]."""

    @property
    def model_name(self) -> str:
        """Return the canonical model name used in persisted measurements.
        Returns:
            The random normal model name.
        """
        return "random-normal-model"

    @property
    def experiment_name(self) -> str:
        """Return the name of the experiment represented by this evaluator.
        Returns:
            The random normal experiment name.
        """
        return "Random Normal Experiment"

    @property
    def model_id(self) -> str:
        """Return the unique model identifier used by the provider.
        Returns:
            The configured random normal model identifier.
        """
        return "random-normal-v1"

    def __init__(
        self,
        mean: float = DEFAULT_MEAN,
        standard_deviation: float = DEFAULT_STANDARD_DEVIATION,
        **parameters: Any,
    ) -> None:
        """Initialize the evaluator with normal distribution parameters.
        Args:
            mean: Mean theta used by the normal distribution.
            standard_deviation: Standard deviation used by the normal
                distribution.
            **parameters: Additional evaluator parameters.
        Raises:
            ValueError: If standard_deviation is negative.
            TypeError: If mean or standard_deviation cannot be converted to float.
        """
        super().__init__(**parameters)
        raw_mean: Any = self._parameter("mean", mean)
        # Mean used to center the generated normal distribution.
        self.mean: float = float(raw_mean)
        raw_standard_deviation: Any = self._parameter(
            "standard_deviation",
            standard_deviation,
        )
        self.standard_deviation: float = float(raw_standard_deviation)

    @property
    def metric_type(self) -> str:
        """Return the continuous variable type used by this evaluator.
        Returns:
            The continuous metric type.
        """
        return "continuous"

    def run(self, **context: Any) -> SamplingObservation | None:
        """Generate one random continuous observation and update evaluator state.
        Args:
            **context: Unused runtime context values.
        Returns:
            A sampling observation containing the random continuous theta and
            token counts.
        """
        del context
        self._theta = min(
            100.0,
            max(0.0, random.gauss(mu=self.mean, sigma=self.standard_deviation)),
        )
        self._prompt_tokens = random.randint(10, 100)
        self._completion_tokens = random.randint(10, 100)
        return SamplingObservation(
            theta=self._theta,
            experiment_name=self.experiment_name,
            model_name=self.model_name,
            model_id=self.model_id,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._completion_tokens + self._prompt_tokens,
        )
```

The evaluator must satisfy these requirements:

- It must inherit from `BaseEvaluator`, as `RandomNormalEvaluator` does in the
  class definition.
- It must expose metadata properties used in the measurements file:
  `model_name`, `experiment_name`, and `model_id`.
- It must expose `metric_type`, usually `"binary"` for pass/fail evaluations or
  `"continuous"` for numeric scores.
- It must implement `run()`. SafeLLM4SE calls this method once per sample, and
  the method must return a `SamplingObservation` with at least `theta`,
  experiment/model metadata, and token counts.
- Its constructor can define any experiment-specific parameters. The example
  accepts `mean` and `standard_deviation`, reads them with `_parameter()`, and
  stores them as typed attributes used later by `run()`.

For example, from the repository root:

```bash
safellm4se-sample --evaluator example_random_evaluator --target-ci-width 10 --mean=60 --standard-deviation=10
```

In this command, `mean` and `standard-deviation` are not interpreted by
SafeLLM4SE itself. SafeLLM4SE converts the option names to constructor
parameters and passes them to the evaluator, so `--standard-deviation` is passed
as `standard_deviation`. A custom evaluator can accept any parameters needed for
the experiment, such as model identifiers, provider settings, benchmark names,
generation temperatures, prompts, or paths to local resources.

### Provider environment variables

The included Gemini and Groq evaluators read their credentials from environment
variables:

```bash
GEMINI_API_KEY="your Gemini API key"
GROQ_API_KEY="your Groq API key"
OLLAMA_HOST="http://localhost:11434"
```

The included Ollama evaluators read the Ollama API base URL from `OLLAMA_HOST`.
If these variables are not defined in the process environment, SafeLLM4SE reads
the same names from a `.env` file in the current working directory. No
command-line parameter is required for these provider settings.

## 2. Reporting one task

Use `safellm4se-report` to filter `measurements.csv` by `task_id` and write a one-row
report with the SafeLLM4SE reporting fields:

```bash
safellm4se-report --input output/measurements.csv --output output/report-qwen-coder.csv --task-id task-id-54 --task-name qwen-coder --boxplot output/qwen-coder-boxplot.svg --violin output/qwen-coder-violin.svg --ecdf output/qwen-coder-ecdf.svg --raincloud output/qwen-coder-raincloud.svg --kde output/qwen-coder-kde.svg
```

Typical console output:

```text
Report written to output\report-qwen-coder.csv.
```

Example report row produced by an example execution of the `qwen2.5-coder:7b` model on the [HumanEval](https://humaneval.org/) benchmark through Ollama:


| date | time | task_id | model_name | model_id | temperature | N | prompt_tokens | completion_tokens | total_tokens | theta_mean | theta_median | theta_min | theta_max | theta_type | sd | cv | iqr | q1 | q3 | ci_method | ci_confidence-level | ci_low | ci_high | ci_width |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-09-02 | 19:09:44 | task-id-54 | qwen-coder | qwen2.5-coder:7b | 2.0 | 30 | 842,430 | 336,100 | 1,178,530 | 0.8272 | 0.8293 | 0.7805 | 0.8659 | continuous | 0.0194 | 2.3496 | 0.0183 | 0.8171 | 0.8354 | t | 95.0 | 0.8200 | 0.8345 | 0.0145 |

The report includes:

- Quality: `theta_mean`, `theta_median`, `theta_min`, `theta_max`.
- Stability: `sd`, `cv`, `iqr`, `q1`, `q3`.
- Uncertainty: `ci_method`, `ci_confidence-level`, `ci_low`, `ci_high`,
  `ci_width`.
- Cost: `prompt_tokens`, `completion_tokens`, `total_tokens`.

## 3. Comparing two tasks

Use `safellm4se-compare` when two task IDs represent two systems, models, or
configurations that should be compared.

Independent design:

```bash
safellm4se-compare --input output/measurements.csv --output output/comparing-independent-deepseek-qwen.csv --task-id-1 task-id-54 --task-name-1 qwen-coder --task-id-2 task-id-56 --task-name-2 deepseek-coder --test-type independent --raincloud output/comparing-deepseek-qwen-raincloud.svg
```

Paired design:

```bash
safellm4se-compare --input output/measurements.csv --output output/comparing-paired-deepseek-qwen.csv --task-id-1 task-id-54 --task-name-1 qwen-coder --task-id-2 task-id-56 --task-name-2 deepseek-coder --test-type paired --boxplot output/comparing-deepseek-qwen-boxplot.svg
```

Typical console output:

```text
Comparison report written to output\compare-independent-deepseek-qwen.csv.
```

Example independent comparison row for an example comparison of the `qwen2.5-coder:7b` and `deepseek-coder:6.7b` models on the [HumanEval](https://humaneval.org/) benchmark through Ollama:

| date | time | task_id_1 | task_id_2 | model_name_1 | model_name_2 | model_id_1 | model_id_2 | temperature_1 | temperature_2 | N_1 | N_2 | total_tokens_1 | total_tokens_2 | theta_mean_1 | theta_mean_2 | test_type | estimated_difference | ci_method | ci_low | ci_high | statistical_test | p_value | effect_size_name | effect_size | effect_size_magnitude |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-09-02 | 18:57:29 | task-id-54 | task-id-56 | qwen-coder | deepseek-coder | qwen2.5-coder:7b | deepseek-coder:6.7b | 2.0 | 2.0 | 30 | 30 | 1,178,530 | 1,611,859 | 0.8272 | 0.6189 | independent | 0.2083 | bootstrap_difference | 0.1937 | 0.2224 | Mann-Whitney U | 2.86e-11 | Cliff's delta | 1.0 | large |

Example plot generated by `safellm4se-compare`:

![Two-sample raincloud plot](../output/compare-deepseek-qwen-raincloud.svg)

## HumanEval examples

One [HumanEval](https://humaneval.org/) problem through Ollama:

```bash
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.ollama.humaneval_oneprogram --task-id ollama-humaneval-1 --n-min 10 --target-ci-width 0.20 --budget-tokens 100000 -- temperature=0.2 problem_number=1 model_id="qwen2.5-coder:7b" model_name="qwen-coder"
```

Full [HumanEval](https://humaneval.org/) benchmark through Groq:

```bash
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.groq.humaneval_fullbench --task-id groq-humaneval-full --n-min 30 --target-ci-width 0.10 --budget-tokens 5000000 -- temperature=2.0 model_id="openai/gpt-oss-20b"
```

Full [HumanEval](https://humaneval.org/) benchmark through Gemini:

```bash
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.gemini.humaneval_fullbench --task-id gemini-humaneval-full --n-min 30 --target-ci-width 0.10 --budget-tokens 5000000 -- temperature=2.0 model_id="gemini-3.1-flash-lite"
```

## Concurrent sampling

Multiple `safellm4se-sample` processes can append to the same output directory. The
sampler uses:

- `.sampling.lock` for process-safe CSV and reservation updates.
- `.sampling_reservations.json` to reserve execution numbers.

If a process crashes after reserving an execution number, the reservation can be
reused after `--reservation-ttl-seconds`.
