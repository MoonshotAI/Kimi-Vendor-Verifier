# [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)

English | [中文](README_zh.md)

Kimi Vendor Verifier (KVV) verifies whether a vendor deployment meets Kimi model accuracy and API feature requirements. The project contains two kinds of evaluation: benchmark evaluations built on [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai), and standalone verifiers that call the chat-completions API directly.

## K3.1 Pro Evaluation Results

| model | thinking effort |
|-------|-----------------|
| kimi-k3.1-pro | max |

| Provider | OCRBench | MMMU Pro Vision | BEAM (1M) | DeepSWE | ToolCall bench |
|----------|----------|-----------------|-----------|---------|----------------|
| Moonshot | 0.872 | 0.841 | 0.3917 | 0.755 | 0.972 |
| AWS | 0.881 | 0.838 | 0.3893 | 0.758 | 0.973 |
| Baseten | 0.881 | 0.847 | 0.3897 | 0.746 | 0.980 |
| Databricks | 0.882 | 0.842 | 0.3972 | 0.750 | 1.0 |
| DigitalOcean | 0.880 | 0.846 | 0.3934 | TBD | 0.980 |
| Fireworks | 0.866 | 0.837 | 0.3888 | 0.754 | 0.986 |
| Inferact (vLLM ref.) | 0.884 | 0.840 | 0.3913 | 0.758 | 0.972 |
| Modal | 0.876 | 0.841 | 0.3883 | 0.764 | 1.0 |
| Nebius | 0.869 | 0.842 | 0.3921 | 0.742 | 0.973 |
| Together | 0.886 | 0.842 | 0.3842 | 0.735 | 0.998 |
| Verda | 0.878 | 0.846 | 0.3976 | 0.757 | 1.0 |

## K3 Evaluation Results

Ranked in the order of submission time.

| model | thinking effort |
|-------|-----------------|
| kimi-k3 | max |

| Provider | OCRBench | MMMU Pro Vision | BEAM (1M) | DeepSWE |
|----------|----------|-----------------|-----------|---------|
| Moonshot | 0.89 | 0.82 | 0.31 | 0.675 |
| Fireworks | 0.89 | 0.82 | 0.3037 | 0.664 |
| Baseten | 0.889 | 0.804 | 0.3219 | 0.693 |
| Together | 0.897 | 0.820 | 0.3160 | 0.678 |
| DigitalOcean | 0.89 | 0.816 | TBD | TBD |
| Inferact (vLLM ref.) | 0.891 | 0.818 | 0.3188 | 0.695 |
| Nebius | 0.878 | 0.814 | 0.2913 | 0.673 |
| Modal | 0.887 | 0.817 | 0.322 | 0.658 |
| RadixArk (SGlang ref.) | 0.895 | 0.820 | 0.3107 | 0.667 |

## Overview

| Type | Task | Description |
|------|------|-------------|
| Inspect-ai benchmark | OCRBench | OCR text recognition benchmark |
| Inspect-ai benchmark | MMMU Pro Vision | Multimodal understanding benchmark (visual QA) |
| Inspect-ai benchmark | AIME 2025 | Mathematical reasoning benchmark; no longer used for K3 |
| Inspect-ai benchmark | Kimi Tool Call | Production badcase regression samples for checking single-step tool-call correction under long context |
| Standalone script | BEAM (1M) | Long-term memory benchmark over 1M-token conversations, see [beam/](beam/README.md) |
| Pytest verifier | `tests/params/` | API parameter constraint pre-flight validation |
| Pytest verifier | `tests/tool_call_json_schema/` | Tool-call argument validation for walle-valid MFJS schemas |
| Pytest verifier | `tests/toolcall_reject_cases/` | Validates malformed tool-call requests are rejected (e.g., missing `function.name`) |
| Pytest verifier | `tests/k3_features/` | K3 feature contract validation, including dynamic tools, response_format, tool_choice, and thinking effort |
| Pytest verifier | `tests/prompt_tokens/` | Verifies vendor-reported `usage.prompt_tokens` against expected constants |
| Agent benchmark | [DeepSWE](https://github.com/datacurve-ai/deep-swe) | Multi-step tool-use and coding-agent evaluation, run on the Pier platform |

## Environment Setup

### Install Dependencies

```bash
uv sync && uv pip install -e .
```

### Configure Environment Variables

Note: the `thinking` parameter has a breaking change in K3, using reasoning_effort instead of thinking; The `ENABLE_THINKING_EFFORT` flag can only be used for K3-series validation. For endpoints that only accept `max_completion_tokens`, set the environment variable `KVV_TOKEN_PARAM=max_completion_tokens` to switch the wire parameter for tests/ testsuites and beam scripts (eval.py use CLI argument, the two spellings never appear together), otherwise using `max_tokens`.

```bash
export KIMI_API_KEY="your-api-key"
export KIMI_BASE_URL="your-base-url"
export ENABLE_THINKING_EFFORT="true"
export KVV_TOKEN_PARAM="max_completion_tokens" #optional
```

Or copy `.env.example` to `.env` and fill in the configuration.

## Pre-flight Validation

Before running benchmarks, complete the API parameter, tool-call schema, K3 feature, and prompt-token pre-flight checks.

### Parameter Constraint Validation: `tests/params/`

Validate that the API correctly constrains immutable parameters such as `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, and `n`.

```bash
# Kimi official API
uv run pytest tests/params --smoke-model kimi/your-model-id --think-mode kimi -v

# Open-source deployment (vLLM/SGLang/KTransformers)
uv run pytest tests/params --smoke-model your-model-id --think-mode opensource -v
```

Run formal benchmarks only after all tests pass.

### Tool Call JSON Schema Validation: `tests/tool_call_json_schema/`

Validate that the vendor can use walle-valid MFJS schemas as tool-call `parameters` and return `tool_calls[].function.arguments` that conform to the schema, and that walle-rejected schemas (e.g. non-terminating recursion) are answered with HTTP 400.

```bash
uv run pytest -n 4 tests/tool_call_json_schema \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    --think-mode "$THINK_MODE" \
    --thinking \
    --reruns 3 \
    --reruns-delay 2 \
    --tool-json-report=tool-call-schema-report.json \
    -ra -v
```

This check runs walle `valid.jsonl` cases, sends each schema as `tools[].function.parameters`, forces a tool call, and uses `jsonschema` to validate the returned `function.arguments` locally. Each selected case is run once with `stream=false` and once with `stream=true`; streaming chunks are reassembled before validation. In GitLab CI, this check runs as the `verify_tool_call_json_schema` job; the same retry options are applied. In addition to the JUnit XML report, a JSON report (`tool-call-schema-report.json`) is produced containing the selected cases, per-case outcomes, and the same `summary` block (total / by_status / by_selection_reason / by_mode) that the original standalone script printed at the end of its log.

The suite also runs supplemental cases from `testdata/walle_validator_cases/go_validator_cases.jsonl` -- schemas from upstream walle Go `validator_test.go` that are absent from the `valid.jsonl` snapshot (see `testdata/walle_validator_cases/SOURCE.md`). Accepted cases send the original schema as `parameters` and validate the returned `function.arguments` against the walle Canonical form; a non-object root schema is wrapped as a `value` parameter before sending. Rejected cases (recursive schemas for which no finite instance exists) assert the endpoint responds with HTTP 400. These supplemental cases are loaded from the fixed path above -- they are not affected by `--case-dir`, `--selection`, or `--max-cases` -- and each runs once with `stream=false` and once with `stream=true`; in the JSON report they appear with `suite` `go_validator_cases` and `selection_reason` `go_validator`.
* Note: MSJS is specific to Kimi's think-mode. Open-source implementations that are not aligned with MSJS will report errors (i.e., fail these checks).

#### Arguments

| Argument | Meaning | Default |
|----------|---------|---------|
| `--smoke-model` | Vendor model name | `MODEL_NAME` env var |
| `--base-url` | API base URL | `KIMI_BASE_URL` env var, otherwise Moonshot API |
| `--api-key` | API key | `KIMI_API_KEY` env var |
| `--case-dir` | walle `validator_cases` directory | `testdata/walle_validator_cases/validator_cases` |
| `--selection` | Case selection mode: `all` / `explicit` / `object` | `all` |
| `--max-cases` | Maximum number of selected cases to run | No limit |
| `--thinking` | Enable thinking mode for the tool-call request | Off |
| `--think-mode` | Thinking parameter format: `kimi`, `opensource` | `THINK_MODE` env var, otherwise `kimi` |
| `--max-tokens` | Maximum output tokens for each tool-call response | `2048` |
| `--tool-json-report` | Path to the additional JSON report artifact | `tool-call-schema-report.json` |

### Tool Call Reject Validation: `tests/toolcall_reject_cases/`

Validates that the vendor correctly rejects malformed tool-call requests, such as those missing the required `function.name` field. This check sends each case in `testdata/toolcall_reject_cases/rejected_case.jsonl` with `stream=false` and asserts that the server responds with `BadRequestError` (HTTP 400).

```bash
uv run pytest tests/toolcall_reject_cases \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    --junitxml=toolcall-reject-report.xml \
    -ra -v
```

The CI job `verify_toolcall_reject` is equivalent to the command above. Locally, run `./run_eval.sh verify_toolcall_result_reject` to execute it in one step.

## K3 Feature Validation: `tests/k3_features/`

Validate K3 API features such as dynamic tools, `response_format`, `tool_choice`, and thinking effort. For endpoints that only accept `max_completion_tokens`, set the environment variable `KVV_TOKEN_PARAM=max_completion_tokens` to switch the wire parameter globally (the two spellings never appear together).

```bash
uv run pytest tests/k3_features \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    -ra -v
```

| Argument | Meaning | Default |
|----------|---------|---------|
| `--smoke-model` | Vendor model name | `MODEL_NAME` env var |
| `--base-url` | API base URL | `KIMI_BASE_URL` env var, otherwise Moonshot API |
| `--api-key` | API key | `KIMI_API_KEY` env var |
| `--think-mode` | `kimi` (official API) or `opensource` (vLLM etc.) | `THINK_MODE` env var, otherwise `kimi` |


With `opensource` (open-source servers such as vLLM):

- Thinking effort is sent via the standard `reasoning_effort` parameter
  instead of the Kimi-private `thinking` object.
- Tests for platform-side (walle/MFJS) request validation and tokenism
  prompt-token groundtruth are skipped -- those belong to the Kimi
  platform layer, not to a bare open-source deployment.

The CI job `verify_k3_features` runs the pytest suite and reports results from the test log.

## Prompt Token Validation: `tests/prompt_tokens/`

Validate that vendor-reported `usage.prompt_tokens` is accurate. This check
sends the text cases in `testdata/prompt_token_cases/cases.jsonl` and vision
cases in `testdata/prompt_token_cases/vision_cases.jsonl` with `stream=true`,
then compares the stream-reported `usage.prompt_tokens` against the expected
constant stored in each case. See
[testdata/prompt_token_cases/README.md](testdata/prompt_token_cases/README.md)
for the case format.

```bash
uv run pytest tests/prompt_tokens \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    -ra -v
```

With `--think-mode opensource`, the suite also sends
`stream_options: {"include_usage": true}` (open-source servers only emit
`usage` in streams when asked) and adjusts the expected constants for
systematic, measured deltas of vLLM serving: +3 for the generation-prefix
channel marker that the platform tokenism counts as pending, and +1 per
image placeholder.

### Why These Tests Are Not in Inspect-ai

`tests/k3_features/`, `tests/prompt_tokens/`, and `tests/tool_call_json_schema/` validate raw chat-completions API behavior, including non-standard `messages[].tools` dynamic-tool declarations, malformed requests that should return HTTP 400, the location of usage fields in streaming chunks, and raw `tools[].function.parameters` JSON schema handling. inspect-ai is better suited for scored benchmark evaluations and may normalize or reject these raw payloads before they reach the vendor endpoint, so these tests remain standalone pytest suites.

## Recommended Benchmark Parameters

### K2.6 and Earlier

| Benchmark | Mode | Temperature | TopP | Max Tokens | Epochs |
|-----------|------|-------------|------|------------|--------|
| OCRBench | Non-Thinking | 0.6 | 0.95 | 16384 | 1 |
| OCRBench | Thinking | 1.0 | 0.95 | 16384 | 1 |
| MMMU | Non-Thinking | 0.6 | 0.95 | 65536 | 1 |
| MMMU | Thinking | 1.0 | 0.95 | 65536 | 1 |
| AIME 2025 | Non-Thinking | 0.6 | 0.95 | 98304 | 32 |
| AIME 2025 | Thinking | 1.0 | 0.95 | 98304 | 32 |

### K2.7

| Benchmark | Mode (thinking, preserve thinking) | Temperature | TopP | Max Tokens | Epochs |
|-----------|------------------------------------|-------------|------|------------|--------|
| OCRBench | Thinking, Keep all | 1.0 | 0.95 | 16384 | 1 |
| MMMU | Thinking, Keep all | 1.0 | 0.95 | 65536 | 1 |
| AIME 2025 | Thinking, Keep all | 1.0 | 0.95 | 98304 | 32 |

### K3

K3 no longer uses AIME 2025 and adds the `reasoning_effort` parameter.

Both kimi-k3 and kimi-k3.1-pro use the following parameters:

| Benchmark | Mode (thinking, preserve thinking, effort) | Temperature | TopP | Max Tokens | Epochs |
|-----------|--------------------------------------------|-------------|------|------------|--------|
| OCRBench | reasoning_effort, low | 1.0 | 0.95 | 16384 | 1 |
| OCRBench | reasoning_effort, high | 1.0 | 0.95 | 16384 | 1 |
| OCRBench | reasoning_effort, max | 1.0 | 0.95 | 16384 | 1 |
| MMMU | reasoning_effort, low | 1.0 | 0.95 | 98304 | 1 |
| MMMU | reasoning_effort, high | 1.0 | 0.95 | 98304 | 1 |
| MMMU | reasoning_effort, max | 1.0 | 0.95 | 98304 | 1 |
| Tool Call | reasoning_effort, low | 1.0 | 0.95 | 32768 | 1 |
| Tool Call | reasoning_effort, high | 1.0 | 0.95 | 32768 | 1 |
| Tool Call | reasoning_effort, max | 1.0 | 0.95 | 32768 | 1 |

## Running Inspect-ai Benchmarks

### OCRBench

#### Non-Thinking

```bash
uv run python eval.py ocrbench --model kimi/your-model-id \
    --think-mode kimi --max-tokens 16384 --stream
```

#### Thinking

```bash
uv run python eval.py ocrbench --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 16384 --stream
```

#### Thinking + reasoning effort (K3)

```bash
uv run python eval.py ocrbench \
    --model "${THINK_MODE}/${MODEL_NAME}" \
    --max-completion-tokens 16384 \
    --think-mode "$THINK_MODE" \
    --stream \
    --max-connections 50 \
    --temperature 1.0 \
    --top-p 0.95 \
    --thinking-effort high
```

### MMMU Pro Vision

#### Non-Thinking

```bash
uv run python eval.py mmmu --model kimi/your-model-id \
    --think-mode kimi --max-tokens 65536 --stream
```

#### Thinking

```bash
uv run python eval.py mmmu --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 65536 --stream
```

#### Thinking + reasoning effort (K3)

```bash
uv run python eval.py mmmu \
    --model "$MODEL" \
    --max-completion-tokens 98304 \
    --think-mode "$THINK_MODE" \
    --stream \
    --max-connections 50 \
    --temperature 1.0 \
    --top-p 0.95 \
    --thinking-effort high
```

### AIME 2025

#### Non-Thinking

```bash
uv run python eval.py aime2025 --model kimi/your-model-id \
    --think-mode kimi --max-tokens 98304 --stream
```

#### Thinking

```bash
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream
```

Run OCRBench first as a quick deployment sanity check, then run the full MMMU and other benchmark evaluations.

### Tool Call

This benchmark uses JSONL data format. See [toolcall_benchmark/README.md](toolcall_benchmark/README.md) for field definitions.

> Note: There is no non-thinking tool-call benchmark. Newer models have no non-thinking requirement, so only the thinking dataset (`toolcall_thinking_samples.stable2.jsonl`) is provided. The commands below are examples — substitute your own model and parameter values.

#### Thinking

```bash
uv run python eval.py kimi_toolcall \
  --model "kimi/${MODEL_NAME}" \
  --max-tokens 32768 \
  --thinking \
  --think-mode "$THINK_MODE" \
  --stream \
  --max-connections 50 \
  --temperature 1.0 \
  --top-p 0.95 \
  --dataset toolcall_benchmark/toolcall_thinking_samples.stable2.jsonl
```

#### Thinking + reasoning effort (K3)

`$THINK_EFFORT` can be `low`, `high`, `max`, or `none` (`none` disables thinking).

```bash
uv run python eval.py kimi_toolcall \
  --model "${THINK_MODE}/${MODEL_NAME}" \
  --max-completion-tokens 32768 \
  --thinking-effort "$THINK_EFFORT" \
  --think-mode "$THINK_MODE" \
  --stream \
  --max-connections 50 \
  --temperature 1.0 \
  --top-p 0.95 \
  --dataset toolcall_benchmark/toolcall_thinking_samples.stable2.jsonl
```

## Agent Benchmark (DeepSWE)

The Tool Call benchmark is built from bad cases previously observed in production. Some samples include repeated tool calls or incorrect tool calls, so this benchmark is mainly useful for checking whether the model can correct a single step under long-context conditions. It is not sufficient to verify whether an agent can solve a task by calling tools multiple times across a full workflow.

To cover that capability, we also add an agent benchmark based on the open-source [DeepSWE](https://github.com/datacurve-ai/deep-swe) benchmark. DeepSWE contains 113 coding-agent tasks and is used here through the open-source Pier platform. The agent runtime uses Kimi-code `0.29.0` or later.

Before running the benchmark on Pier, register `kimi-code` on the Pier agent first. See the Kimi-code `0.29.0` setup reference: https://github.com/MoonshotAI/kimi-code/tree/%40moonshot-ai/kimi-code%400.29.0

## Inspect-ai Parameters

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `benchmark` | Evaluation task: `ocrbench`, `mmmu`, `aime2025`, `kimi_toolcall` | `ocrbench` |
| `--model` | Model identifier, for example `kimi/your-model-id` | **Required** |
| `--max-tokens` | Maximum output tokens, sent as `max_tokens`; mutually exclusive with `--max-completion-tokens` | One of the two is required |
| `--max-completion-tokens` | Equivalent of `--max-tokens`; for endpoints that reject `max_tokens` | One of the two is required |
| `--thinking` | Enable thinking mode (K2 series). K3 remains compatible, but using `reasoning_effort` via `--thinking-effort` is recommended | Off |
| `--think-mode` | Thinking parameter format: `kimi` or `opensource` | `kimi` |
| `--temperature` | Sampling temperature | Optional; determined by server or model defaults if omitted |
| `--top-p` | Top-p sampling | Optional; determined by server or model defaults if omitted |
| `--stream` | Enable streaming, recommended for long reasoning requests | Off |
| `--max-connections` | Maximum concurrent connections | Per benchmark |
| `--epochs` | Number of sampling epochs | Per benchmark |
| `--client-timeout` | HTTP timeout in seconds | `86400` |
| `--thinking-effort` | Thinking effort: `low`, `high`, `max`, or `none`. With `--think-mode kimi` it is sent directly as `reasoning_effort` (`none` disables thinking, equivalent to non-thinking); no `--thinking` flag needed | None |
| `--dataset` | Only used for the `kimi_toolcall` benchmark | None |

### Thinking Mode Parameters

With `--think-mode kimi`, when `--thinking-effort` is given, both the legacy `thinking` object (with `keep`/`effort`) and the top-level `reasoning_effort` are sent — the two forms coexist (K2-compatible stacks read the former, K3 reads the latter). `none` disables thinking in both forms. An explicit effort implies thinking is enabled, so `--thinking` can be omitted. The same applies to K3 open-source deployments (`--think-mode opensource`), where the legacy form is `chat_template_kwargs`.

| Model Type | Parameter Combination | `extra_body` Sent |
|------------|-----------------------|-------------------|
| Kimi official K2 series + thinking off | `--think-mode kimi` | `{"thinking": {"type": "disabled"}}` |
| Kimi official K2 series + thinking on | `--thinking --think-mode kimi` | `{"thinking": {"type": "enabled"}}` |
| Kimi official K3 series + reasoning effort low | `--think-mode kimi --thinking-effort low` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "low"}}` |
| Kimi official K3 series + reasoning effort high | `--think-mode kimi --thinking-effort high` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "high"}}` |
| Kimi official K3 series + reasoning effort max | `--think-mode kimi --thinking-effort max` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "max"}}` |
| Kimi official K3 series + thinking off (effort `none`) | `--think-mode kimi --thinking-effort none` | `{"thinking": {"type": "disabled"}}` |
| Open-source framework K2 series + thinking off | `--think-mode opensource` | `{"chat_template_kwargs": {"thinking": false}}` |
| Open-source framework K2 series + thinking on | `--thinking --think-mode opensource` | `{"chat_template_kwargs": {"thinking": true}}` |
| Open-source framework K3 series + reasoning effort low | `--think-mode opensource --thinking-effort low` | `{"reasoning_effort": "low"}` |
| Open-source framework K3 series + reasoning effort high | `--think-mode opensource --thinking-effort high` | `{"reasoning_effort": "high"}` |
| Open-source framework K3 series + reasoning effort max | `--think-mode opensource --thinking-effort max` | `{"reasoning_effort": "max"}` |

### Viewing Results

```bash
# View logs with inspect view
uv run inspect view

# Logs are saved in the logs/ directory
```

Each model call in the `.eval` log now includes a `request_context` object
containing `client_request_id`, `request_id`, and `trace_id` (when returned by
the upstream service). You can find it in:

- `samples/<id>.json` → `metadata.request_context` (shown in the **Metadata** panel of `inspect view`)
- `samples/<id>.json` → `output.metadata.request_context`
- `samples/<id>.json` → `events[*].call.response._request_context`

### Resuming an Interrupted Benchmark

```bash
uv run inspect eval-retry logs/<log-file>.eval
```

## Notes

### AIME 2025

AIME evaluation produces many output tokens; please note the following:

1. Timeout settings: the client default is `--client-timeout 86400` (24 hours), and the server, gateway, or proxy timeout should also be long enough.
2. Streaming: strongly recommended with `--stream`; non-streaming requests are more prone to timeout in thinking mode.
3. Concurrency control: if you see many 429s or `RemoteProtocolError`s, lower `--max-connections`.
4. Quick validation: first run all samples with `--epochs 1`, then run the full evaluation after configuration is confirmed.

```bash
# Step 1: Quick validation (30 samples x 1 epoch)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream --epochs 1

# Step 2: Full evaluation (30 samples x 32 epochs)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream
```

### Automatic Retry Mechanism

The following network-related errors are automatically retried with exponential backoff (1-60 seconds), no manual configuration needed:

| Error Type | Description |
|------------|-------------|
| `RateLimitError` / `429` | Server-side rate limiting |
| `APIConnectionError` | Connection failure |
| `ReadError` / `RemoteProtocolError` | Network read error |

Non-network errors, such as model output format issues, are not retried and are logged directly for later analysis.

## Project Structure

```text
├── eval.py              # Main evaluation CLI entrypoint
├── tests/params/        # Pre-flight parameter validation
├── tests/tool_call_json_schema/ # walle tool-call schema validation
├── tests/toolcall_reject_cases/ # malformed tool-call request rejection validation
├── tests/prompt_tokens/ # usage.prompt_tokens accuracy validation
├── tests/k3_features/   # K3 feature contract validation
├── kimi_model.py        # Kimi Model API implementation
├── aime2025.py          # AIME 2025 evaluation task
├── mmmu_pro_vision.py   # MMMU Pro Vision evaluation task
├── ocr_bench.py         # OCRBench evaluation task
├── kimi_toolcall.py     # Kimi Tool Call evaluation task
├── testdata/            # JSON schema and prompt-token test data
├── toolcall_benchmark/  # Kimi Tool Call test data, with field definitions in the directory README
├── beam/                # BEAM 1M context data and scripts, see the directory README
├── logs/                # Evaluation logs
└── pyproject.toml       # Project configuration
```

## Contact

If you have any questions or suggestions, please contact contact-kvv@kimi.com.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
