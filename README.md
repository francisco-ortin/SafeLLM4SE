# SafeLLM4SE

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Latest release](https://img.shields.io/github/v/release/francisco-ortin/SafeLLM4SE?include_prereleases)](https://github.com/francisco-ortin/SafeLLM4SE/releases)
[![PyPI](https://img.shields.io/pypi/v/safellm4se)](https://img.shields.io/pypi/v/safellm4se)

SafeLLM4SE is a Python toolkit for statistically principled evaluation of
LLM-based software engineering systems. It treats each LLM execution as a
sample from a stochastic process, then reports quality, stability, uncertainty,
resource usage, and statistical comparisons instead of relying on a single run.

The project exposes three command-line programs:

- `safellm4se-sample`: runs adaptive sampling with a user-selected evaluator.
- `safellm4se-report`: summarizes one sampled task into a SafeLLM4SE report CSV.
- `safellm4se-compare`: compares two sampled tasks with the SafeLLM4SE comparison protocol.

## Install

SafeLLM4SE is available on PyPI as `safellm4se`, so you only need to run:

```bash
pip install "SafeLLM4SE[all]"
```

If you want to install SafeLLM4SE from its source code, check the [installation details](doc/installation.md).

## Quick Start

SafeLLM4SE collects repeated observations with `safellm4se-sample`.
Then, it can be used to report or visualize one sample of observations with `safellm4se-report`,
or compare task samples with `safellm4se-compare`.

### Sampling

First, you need to generate a sample of repeated observations.
For this purpose, `safellm4se-sample` loads an evaluator class and repeatedly calls it until a stopping condition is met
(maximum number of tokens consumed or the confidence interval width is below a threshold).
You commonly implement the process being measured as an evaluator,
but we provide several example evaluators, including default implementations for [Ollama](https://ollama.com/),
[Gemini](https://ai.google.dev/gemini-api/docs), and [Groq](https://console.groq.com/) APIs.
The Gemini and Groq evaluators read `GEMINI_API_KEY` and `GROQ_API_KEY` from
the process environment. If they are not set, SafeLLM4SE reads them from a
`.env` file in the current working directory.
The Ollama evaluators read the API host from `OLLAMA_HOST` in the same way.

For this example, we perform adaptive sampling on a random evaluator to generate two different samples:

```bash
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.random_normal_evaluator --mean=60 --standard-deviation=20
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.random_normal_evaluator --mean=40 --standard-deviation=10 
```

### Reporting

If you have already created a sample of repeated observations with the task identifier `task-id-1`,
stored in `output/measurements.csv`,
you can create reports and visualizations with `safellm4se-report`.
The report includes the sample size, token usage, central tendency, variability, and confidence interval information.
The supported visualizations are boxplot, violin plot, empirical cumulative distribution function (ECDF), raincloud plot, and kernel density estimate (KDE).

Let's generate a report for two different task identifiers, `task-id-1` and `task-id-2`, with boxplot and KDE visualizations:

```bash
safellm4se-report --input output/measurements.csv --output output/report-demo.csv --task-id task-id-1 --boxplot output/demo-boxplot.svg
safellm4se-report --input output/measurements.csv --output output/report-demo.csv --task-id task-id-2 --kde output/demo-kde.svg
```

Example CSV report generated with `safellm4se-report` for the LLMs `qwen2.5-coder:7b` and `deepseek-coder:6.7b` running [HumanEval](https://humaneval.org/):


| task_id | model_name | model_id | N | total_tokens | theta_mean | sd | ci_method | ci_low | ci_high | ci_width |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| task-id-54 | qwen-coder | qwen2.5-coder:7b | 30 | 1,178,530 | 0.8272 | 0.0194 | t | 0.8200 | 0.8345 | 0.0145 |

### Comparing

You can also compare two samples of repeated observations with `safellm4se-compare`.
The report will tell you the estimated difference, confidence interval, statistical test,
p-value, and effect size.
It also generates figures comparing the two samples, including the visualizations mentioned in
[reporting](#reporting).

Let's compare the two samples with task identifiers `task-1` and `task-1` using an *independent* test 
(adaptive sampling may generate different number of observations for each sample), and generate a raincloud plot:

```bash
safellm4se-compare --input output/measurements.csv --output output/comparing-demo.csv --task-id-1 task-id-1 --task-id-2 task-id-2 --test-type independent --raincloud output/comparing-demo-raincloud.svg
```

Example plot generated with `safellm4se-compare` for two different models (`qwen2.5-coder:7b` and `deepseek-coder:6.7b`) running [HumanEval](https://humaneval.org/):

![Two-sample raincloud plot](output/compare-deepseek-qwen-raincloud.svg)

The following CSV comparison report has also been generated:

| task_id_1 | task_id_2 | test_type | estimated_difference | ci_low | ci_high | statistical_test | p_value | effect_size_name | effect_size | effect_size_magnitude |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| task-id-54 | task-id-56 | independent | 0.2083 | 0.1937 | 0.2224 | Mann-Whitney U | 2.86e-11 | Cliff's delta | 1.0 | large |

## Documentation

- [Methodology](doc/methodology.md)
- [Installation](doc/installation.md)
- [Usage Guide](doc/usage.md)
- [CLI Reference](doc/cli-reference.md)
- [Evaluator Guide](doc/evaluators.md)

## License

See [LICENSE](LICENSE).
