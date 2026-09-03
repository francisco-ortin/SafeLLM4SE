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

The sampler appends rows to `output/measurements.csv` (no existing rows are overwritten). Example row produced by the current project data:

```csv
date,time,task_id,experiment_name,model_name,model_id,execution_number,prompt_tokens,completion_tokens,total_tokens,theta,metric_type,evaluator,evaluator_parameters,run_id,temperature
2026-09-02,11:15:35,task-id-54,ollama-humaneval-fullbench,qwen-coder,qwen2.5-coder:7b,1,28081,11703,39784,0.8475609756097561,continuous,OllamaHumanEvalFullBenchEvaluator,"{""temperature"": 2.0}",16835a42-013b-44fc-ba73-002f9b0cb0f4,2.0
```

### Passing evaluator parameters

Arguments after the CLI separator `--` are passed to the evaluator constructor.
Both forms are accepted:

```bash
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.random_normal_evaluator -- mean=60 standard_deviation=10
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.random_normal_evaluator -- --mean=60 --standard-deviation=10
```

Values are converted with Python literal syntax when possible, so booleans,
numbers, `None`, lists, and dictionaries can be passed directly.

The values of these *free* parameters are ignored by SafeLLM4SE, but the evaluator can use them to configure its behavior.
Their values are passed to the evaluator constructor so the evaluator can use them.
In this way, the evaluator can be configured to run different experiments with the same evaluator class.
One typical use of this feature is to pass the `temperature` parameter to the evaluator, which configures the LLM model temperature.

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

Example report row produced by the current project data:

```csv
date,time,task_id,model_name,model_id,temperature,N,prompt_tokens,completion_tokens,total_tokens,theta_mean,theta_median,theta_min,theta_max,theta_type,sd,cv,iqr,q1,q3,ci_method,ci_confidence-level,ci_low,ci_high,ci_width
2026-09-02,19:09:44,task-id-54,qwen-coder,qwen2.5-coder:7b,2.0,30,842430,336100,1178530,0.8272357723577236,0.8292682926829268,0.7804878048780488,0.8658536585365854,continuous,0.019436705668000667,2.349596852249712,0.018292682926829285,0.8170731707317073,0.8353658536585366,t,95.0,0.8199779871829312,0.834493557532516,0.014515570349584728
```

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

Example independent comparison row from the current project data:

```csv
date,time,task_id_1,task_id_2,model_name_1,model_name_2,model_id_1,model_id_2,temperature_1,temperature_2,N_1,N_2,total_tokens_1,total_tokens_2,theta_mean_1,theta_mean_2,test_type,estimated_difference,ci_method,ci_low,ci_high,statistical_test,p_value,effect_size_name,effect_size,effect_size_magnitude
2026-09-02,18:57:29,task-id-54,task-id-56,qwen-coder,deepseek-coder,qwen2.5-coder:7b,deepseek-coder:6.7b,2.0,2.0,30,30,1178530,1611859,0.8272357723577236,0.6189024390243902,independent,0.20833333333333337,bootstrap_difference,0.19369410569105683,0.2223628048780488,Mann-Whitney U,2.8591961948613224e-11,Cliff's delta,1.0,large
```

Example plot generated by `safellm4se-compare`:

![Two-sample raincloud plot](../output/compare-deepseek-qwen-raincloud.svg)

## HumanEval examples

One [HumanEval](https://humaneval.org/) problem through Ollama:

```bash
safellm4se-sample --evaluator safellm4se.sampling.myevaluators.ollama.humaneval_oneprogram --task-id ollama-humaneval-1 --n-min 10 --target-ci-width 0.20 --budget-tokens 100000 -- temperature=0.2 problem_number=1 ollama_host=http://localhost:11434 model_id="qwen2.5-coder:7b" model_name="qwen-coder"
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
