# Methodology

SafeLLM4SE is based on the methodology described in the article
*SafeLLM4SM: Statistical Evaluation and Reporting for LLM-Based Software Engineering Systems*.

The core idea is that an LLM-based software engineering system should not be evaluated
as if a single output represented deterministic behavior. Each execution is a
realization of a stochastic process.

For a prompt, model, and configuration, repeated executions produce observed
metric values. SafeLLM4SE calls each numeric observed value $\theta$ (`theta`) and estimates
the behavior of the underlying process from a sample of repeated observations.

## What SafeLLM4SE reports

SafeLLM4SE separates three concepts that are often conflated:

| Concept | Meaning | Current report fields |
|---|---|---|
| Quality | Central performance under the evaluated conditions. | `theta_mean`, `theta_median`, `theta_min`, `theta_max` |
| Stability | Variability across repeated executions. | `sd`, `cv`, `iqr`, `q1`, `q3` |
| Estimation uncertainty | Precision of the finite-sample estimate. | `ci_method`, `ci_confidence-level`, `ci_low`, `ci_high`, `ci_width` |

The reports also include model/configuration identifiers and token usage so the
evaluation remains reproducible and cost-aware.

## Adaptive sampling

Instead of choosing a fixed number of executions in advance, SafeLLM4SE samples
adaptively. A run starts collecting observations and stops only after the
minimum sample size has been reached and one of these conditions is true:

- the confidence interval width is no larger than the target width; or
- the token budget has been reached.

This makes the number of executions depend on observed variability. Stable
systems can stop earlier, while highly variable systems require more evidence.

The main parameters are:

| Parameter | Meaning |
|---|---|
| `--n-min` | Minimum number of observations before stopping is allowed. |
| `--target-ci-width` | Maximum accepted total confidence interval width. |
| `--budget-tokens` | Token budget for the task/experiment/model key. |
| `--confidence-level` | Confidence level used for interval estimation. |

## Confidence intervals

SafeLLM4SE selects confidence interval methods by metric type:

| Metric type | Method |
|---|---|
| `binary` | Wilson score interval |
| `continuous` with `--ci-method t` | Student t interval |
| `continuous` with `--ci-method bootstrap` | Percentile bootstrap interval |
| `continuous` with `--ci-method auto` | Normality check, then t or bootstrap |

SciPy improves normality checks and t critical values. Without SciPy, the code
uses standard-library fallbacks.

## Statistical comparison

`safellm4se-compare` implements the SafeLLM4SE comparison protocol for two task samples:

| Design | Significance test | Difference CI | Effect size |
|---|---|---|---|
| Independent | Mann-Whitney U | Bootstrap CI for `theta_mean_1 - theta_mean_2` | Cliff's delta |
| Paired | Wilcoxon signed-rank | Paired bootstrap CI for `theta_mean_1 - theta_mean_2` | Matched-pairs rank-biserial correlation |

The comparison output should be interpreted as statistical evidence, not as an
automatic ranking rule. A useful decision should consider the estimated
difference, its confidence interval, statistical evidence, effect size, and the
stability of both systems.

## HumanEval case study

The repository includes example [HumanEval](https://humaneval.org/) evaluators for [Ollama](https://ollama.com/), Gemini, and
Groq. The full-benchmark evaluators run all 164 [HumanEval](https://humaneval.org/) problems in each
sampling observation and report `theta` as the proportion of problems solved in
that complete execution.

The current `output/` directory contains example reports and plots comparing
`qwen2.5-coder:7b` and `deepseek-coder:6.7b` through Ollama-backed [HumanEval](https://humaneval.org/)
runs.
