# [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)

[English](README.md) | 中文

## K3.1 Pro 评测结果

| model | thinking effort |
|-------|-----------------|
| kimi-k3.1-pro | max |

| Provider | OCRBench | MMMU Pro Vision | BEAM (1M) | DeepSWE | ToolCall bench |
|----------|----------|-----------------|-----------|---------|----------------|
| Moonshot | 0.872 | 0.841 | 0.3917 | 0.755 | 0.972 |
| Baseten | 0.881 | 0.847 | 0.3897 | 0.746 | 0.980 |
| DigitalOcean | 0.880 | 0.846 | TBD | TBD | 0.980 |
| Fireworks | 0.866 | 0.837 | 0.3888 | 0.754 | 0.986 |
| Inferact | 0.884 | TBD | TBD | TBD | 0.972 |
| Modal | 0.876 | 0.841 | 0.3883 | 0.764 | 1.0 |
| Nebius | 0.869 | 0.842 | 0.3921 | TBD | 0.973 |
| Together | 0.886 | 0.842 | 0.3842 | 0.735 | 0.998 |
| Verda | 0.878 | 0.846 | 0.3976 | 0.757 | 1.0 |

## K3 评测结果
榜单按照提交测试的时间顺序排列

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




## 概览

| 类型 | 任务 | 说明 |
|------|------|------|
| Inspect-ai benchmark | OCRBench | OCR 文字识别能力评测 |
| Inspect-ai benchmark | MMMU Pro Vision | 多模态理解评测（视觉问答） |
| Inspect-ai benchmark | AIME 2025 | 数学推理能力评测；K3 不再使用 |
| Inspect-ai benchmark | Kimi Tool Call | badcase 回归样本，用于验证长上下文下单步 tool-call 纠错能力 |
| 独立脚本 | BEAM (1M) | 1M token 长对话长期记忆评测，见 [beam/](beam/README_zh.md) |
| Pytest verifier | `tests/params/` | API 参数约束预检 |
| Pytest verifier | `tests/tool_call_json_schema/` | walle-valid MFJS schema 的 tool-call 参数校验 |
| Pytest verifier | `tests/toolcall_reject_cases/` | 验证畸形 tool-call 请求是否被正确拒绝（例如缺少 `function.name`） |
| Pytest verifier | `tests/k3_features/` | K3 feature contract 校验，包括 dynamic tools、response_format、tool_choice、thinking effort |
| Pytest verifier | `tests/prompt_tokens/` | 验证 vendor 上报的 `usage.prompt_tokens` 是否与期望常量一致 |
| Agent benchmark | [DeepSWE](https://github.com/datacurve-ai/deep-swe) | 多步 tool 使用和 coding-agent 能力评测，使用 Pier 平台评测 |


## 环境准备

### 安装依赖

```bash
uv sync && uv pip install -e .
```

### 配置环境变量
注意，thinking 参数在 K3 有 breaking change，使用 reasoning_effort 替代 thinking；ENABLE_THINKING_EFFORT 标记只能用于 K3 系列验证。环境变量KVV_TOKEN_PARAM 作用于 tests/套件与 beam 脚本(eval.py 走 CLI 参数，不读这个变量), 不设置这个环境变量tests测试套件与beam默认发送 `max_tokens`；（两种写法不会同时出现）。
```bash
export KIMI_API_KEY="your-api-key"
export KIMI_BASE_URL="your-base-url"
export ENABLE_THINKING_EFFORT="true"
export KVV_TOKEN_PARAM="max_completion_tokens"
```

或复制 `.env.example` 到 `.env` 并填入配置。

## 预检验证

在运行 benchmark 之前，建议先完成 API 参数、tool-call schema、K3 feature 和 prompt token 相关预检。

### 参数约束验证：`tests/params/`

验证 API 是否正确约束不可变参数，例如 `temperature`、`top_p`、`presence_penalty`、`frequency_penalty` 和 `n`。

```bash
# Kimi 官方 API
uv run pytest tests/params --smoke-model kimi/your-model-id --think-mode kimi -v

# 开源部署（vLLM/SGLang/KTransformers）
uv run pytest tests/params --smoke-model your-model-id --think-mode opensource -v
```

所有测试通过后，再运行正式 benchmark。

### Tool Call JSON Schema 验证：`tests/tool_call_json_schema/`

验证 vendor 能否把 walle-valid MFJS schema 作为 tool-call `parameters` 使用，并返回符合 schema 的 `tool_calls[].function.arguments`；同时验证 walle 拒绝的 schema（如无法终止的递归）会被端点以 HTTP 400 拒绝。

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

该检查运行 walle `valid.jsonl` 用例，把每个 schema 作为 `tools[].function.parameters` 发送，强制触发 tool call，并使用 `jsonschema` 在本地校验返回的 `function.arguments`。每条 case 会分别用 `stream=false` 和 `stream=true` 各发一次，streaming 分片会在校验前重新组装。命令中的 `--reruns` 和 `--reruns-delay` 会自动重试失败的 case，最大重试次数 3 次，重试间隔 2 秒。除 JUnit XML 报告外，还会生成一份 JSON 报告（`tool-call-schema-report.json`），包含已选 case、每条 case 的结果，以及与原独立脚本末尾相同的 `summary` 汇总（total / by_status / by_selection_reason / by_mode）。

此外，该套件还运行 `testdata/walle_validator_cases/go_validator_cases.jsonl` 中的补充用例——来自上游 walle Go `validator_test.go`、但不在 `valid.jsonl` 快照中的 schema（见 `testdata/walle_validator_cases/SOURCE.md`）。accepted 用例把原始 schema 作为 `parameters` 发送，并针对 walle Canonical 形式校验返回的 `function.arguments`；非 object 根 schema 会先包装为 `value` 参数再发送。rejected 用例（不存在有限实例的递归 schema）断言端点返回 HTTP 400。这些补充用例从上述固定路径加载，不受 `--case-dir`、`--selection`、`--max-cases` 影响，每条分别以 `stream=false` 和 `stream=true` 各运行一次；在 JSON 报告中以 `suite` 为 `go_validator_cases`、`selection_reason` 为 `go_validator` 记录。
* 注意 ：MSJS是针对kimi think-mode, opensource的实现没有对齐MSJS会有报错。

#### 参数说明

所有测试套件默认发送 `max_tokens`。对于只接受 `max_completion_tokens` 的端点，设置环境变量 `KVV_TOKEN_PARAM=max_completion_tokens` 即可全局切换线上参数（两种写法不会同时出现）。

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--smoke-model` | vendor 模型名称 | 环境变量 `MODEL_NAME` |
| `--base-url` | API base URL | 环境变量 `KIMI_BASE_URL`，否则 Moonshot API |
| `--api-key` | API key | 环境变量 `KIMI_API_KEY` |
| `--case-dir` | walle `validator_cases` 目录 | `testdata/walle_validator_cases/validator_cases` |
| `--selection` | case 选择模式：`all` / `explicit` / `object` | `all` |
| `--max-cases` | 最多运行多少个已选 case | 不限制 |
| `--thinking` | 开启 tool-call 请求的 thinking mode | 关闭 |
| `--think-mode` | thinking 参数格式：`kimi`、`opensource` | 环境变量 `THINK_MODE`，否则 `kimi` |
| `--max-tokens` | 每次 tool-call response 的最大输出 token 数 | `2048` |
| `--tool-json-report` | 额外的 JSON 报告产物路径 | `tool-call-schema-report.json` |

### Tool Call Reject 验证：`tests/toolcall_reject_cases/`

验证 vendor 能否正确拒绝缺少 `function.name` 等必填字段的畸形 tool-call 请求。该检查把 `testdata/toolcall_reject_cases/rejected_case.jsonl` 中的每条 case 以 `stream=false` 发送，并断言服务端返回 `BadRequestError`（HTTP 400）。

```bash
uv run pytest tests/toolcall_reject_cases \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    --junitxml=toolcall-reject-report.xml \
    -ra -v
```

CI job `verify_toolcall_reject` 与本命令等价；本地可用 `./run_eval.sh verify_toolcall_result_reject` 一键执行。

## K3 Feature 验证：`tests/k3_features/`

验证 K3 新功能 API，例如 dynamic tools、`response_format`、`tool_choice` 和 thinking effort。所有测试套件默认发送 `max_tokens`。对于只接受 `max_completion_tokens` 的端点，设置环境变量 `KVV_TOKEN_PARAM=max_completion_tokens` 即可全局切换线上参数（两种写法不会同时出现）。

```bash
uv run pytest tests/k3_features \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    -ra -v
```

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--smoke-model` | Vendor model name | 环境变量 `MODEL_NAME` |
| `--base-url` | API base URL | 环境变量 `KIMI_BASE_URL`，否则 Moonshot API |
| `--api-key` | API key | 环境变量 `KIMI_API_KEY` |
| `--think-mode` | `kimi`（官方 API）或 `opensource`（vLLM 等） | 环境变量 `THINK_MODE`，否则 `kimi` |

使用`opensource`时：
- thinking effort 通过标准 `reasoning_effort` 参数传递，而非 Kimi 私有的 `thinking` 对象。
- 平台侧（walle/MFJS）请求校验与 tokenism prompt token 基准的用例会跳过——这些属于 Kimi 平台层，不属于裸开源部署。

CI job `verify_k3_features` 会运行 pytest 套件并从测试日志中报告结果。

## Prompt Token 验证：`tests/prompt_tokens/`

验证 vendor 上报的 `usage.prompt_tokens` 是否准确。该检查把 `testdata/prompt_token_cases/cases.jsonl` 中的每条 case 以 `stream=true` 发送，并将流中上报的 `usage.prompt_tokens` 与 case 中存储的期望常量做精确比对。case 格式见 [testdata/prompt_token_cases/README.md](testdata/prompt_token_cases/README.md)。

```bash
uv run pytest tests/prompt_tokens \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    -ra -v
```

使用 `--think-mode opensource` 时，套件会额外发送 `stream_options: {"include_usage": true}`（开源 server 只在显式请求时才在流中返回 `usage`），并按 vLLM 的系统性、已实测的差异修正期望常量：+3（generation prefix 尾部的通道开启标记，平台 tokenism 计为 pending）、+1/张图片占位符。


### 为什么tests 不放在 inspect-ai

`tests/k3_features/` ，`tests/prompt_tokens/`， `tests/tool_call_json_schema` 验证的是原始 chat-completions API 行为，包括非标准的 `messages[].tools` dynamic-tool 声明、预期返回 HTTP 400 的 malformed requests，以及 streaming chunk 中的 usage 上报位置。inspect-ai 更适合带 scorer 的 benchmark 行为评测，并且可能会在请求到达 vendor endpoint 前规范化或拒绝这些 raw payload，因此这些tests保持为独立脚本。

## benchmark评测推荐参数配置

### K2.6 及以前

| Benchmark | 模式 | Temperature | TopP | Max Tokens | Epochs |
|-----------|------|-------------|------|------------|--------|
| OCRBench | Non-Thinking | 0.6 | 0.95 | 8192 | 1 |
| OCRBench | Thinking | 1.0 | 0.95 | 16384 | 1 |
| MMMU | Non-Thinking | 0.6 | 0.95 | 16384 | 1 |
| MMMU | Thinking | 1.0 | 0.95 | 65536 | 1 |
| AIME 2025 | Non-Thinking | 0.6 | 0.95 | 16384 | 32 |
| AIME 2025 | Thinking | 1.0 | 0.95 | 98304 | 32 |

### K2.7

| Benchmark | 模式（thinking, preserve thinking） | Temperature | TopP | Max Tokens | Epochs |
|-----------|--------------------------------------|-------------|------|------------|--------|
| OCRBench | Thinking, Keep all | 1.0 | 0.95 | 16384 | 1 |
| MMMU | Thinking, Keep all | 1.0 | 0.95 | 65536 | 1 |
| AIME 2025 | Thinking, Keep all | 1.0 | 0.95 | 98304 | 32 |

### K3

K3 不再使用 AIME 2025，并增加 `reasoning_effort` 参数。

kimi-k3、kimi-k3.1-pro 都使用如下参数：

| Benchmark | 模式（thinking, preserve thinking, effort） | Temperature | TopP | Max Tokens | Epochs |
|-----------|---------------------------------------------|-------------|------|------------|--------|
| OCRBench | reasoning_effort, low | 1.0 | 0.95 | 16384 | 1 |
| OCRBench | reasoning_effort, high | 1.0 | 0.95 | 16384 | 1 |
| OCRBench | reasoning_effort, max | 1.0 | 0.95 | 16384 | 1 |
| MMMU | reasoning_effort, low | 1.0 | 0.95 | 98304 | 1 |
| MMMU | reasoning_effort, high | 1.0 | 0.95 | 98304 | 1 |
| MMMU | reasoning_effort, max | 1.0 | 0.95 | 98304 | 1 |
| Tool Call | reasoning_effort, low | 1.0 | 0.95 | 32768 | 1 |
| Tool Call | reasoning_effort, high | 1.0 | 0.95 | 32768 | 1 |
| Tool Call | reasoning_effort, max | 1.0 | 0.95 | 32768 | 1 |

## 运行 Inspect-ai Benchmarks

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

#### Thinking + reasoning effort（K3）

```bash
uv run python eval.py ocrbench \
    --model "$MODEL" \
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

#### Thinking + reasoning effort（K3）

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

建议先运行 OCRBench 快速验证部署是否正确，确认通过后再运行 MMMU等完整评测。

### Tool Call

该 benchmark 采用 JSONL 数据格式，字段说明见 [toolcall_benchmark/README_zh.md](toolcall_benchmark/README_zh.md)。

> 注：没有 non-thinking 的 tool call benchmark。后续模型没有 non-thinking 需求，因此只提供 thinking 数据集。以下命令仅为示例，请按实际情况替换模型与参数。

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

#### Thinking + reasoning effort（K3）

`$THINK_EFFORT` 可以是 `low`、`high`、`max`。

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

## Agent Benchmark（DeepSWE）

Tool Call benchmark 来源于线上曾经出现的 badcases。其中部分样本会出现重复调用 tool 或错误调用 tool 的情况，因此该 benchmark 主要用于验证模型在长上下文场景下，能否对单个 step 中的错误进行纠正。它不足以验证 agent 是否具备通过多次调用 tool 完整解决问题的能力。

为覆盖这类能力，我们额外增加了基于开源 [DeepSWE](https://github.com/datacurve-ai/deep-swe) 的 agent benchmark。DeepSWE 共 113 道 coding-agent 题目，这里使用开源 Pier 平台运行。agent runtime 使用 Kimi-code `0.29.0`及以上版本。

使用 Pier 平台运行 benchmark 前，需要先将 `kimi-code` 注册到 Pier agent 上。Kimi-code `0.29.0` 的配置参考：https://github.com/MoonshotAI/kimi-code/tree/%40moonshot-ai/kimi-code%400.29.0

## Inspect-ai 参数说明

### 可用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `benchmark` | 评测任务：`ocrbench`、`mmmu`、`aime2025`、`kimi_toolcall` | `ocrbench` |
| `--model` | 模型标识，例如 `kimi/your-model-id` | **必填** |
| `--max-tokens` | 最大输出 token 数，以 `max_tokens` 发送；与 `--max-completion-tokens` 互斥 | 两者必填其一 |
| `--max-completion-tokens` | 与 `--max-tokens` 等价， 用于拒绝 `max_tokens` 的端点 | 两者必填其一 |
| `--thinking` | 开启思考模式，用于K2系列, K3兼容，但建议直接使用reasoning_effort | 关闭 |
| `--think-mode` | 思考参数格式：`kimi` 或 `opensource` | `kimi` |
| `--temperature` | 采样温度 | 可不设置，由服务端或模型默认值决定 |
| `--top-p` | Top-p 采样 | 可不设置，由服务端或模型默认值决定 |
| `--stream` | 启用流式传输，推荐用于长推理请求 | 关闭 |
| `--max-connections` | 最大并发连接数 | 按 benchmark |
| `--epochs` | 采样次数 | 按 benchmark |
| `--client-timeout` | HTTP 超时时间，单位秒 | `86400` |
| `--thinking-effort` | 思考档位：`low`、`high`、`max` 或 `none`。`--think-mode kimi` 时直接以 `reasoning_effort` 发送（`none` 表示关闭思考，等价于 non-thinking），无需再传 `--thinking` | 无 |
| `--dataset` | 仅用于 `kimi_toolcall` benchmark | 无 |

### 思考模式参数

`--think-mode kimi` 时，只要传了 `--thinking-effort` 就会同时发送旧版 `thinking` 对象（含 `keep`/`effort`）和顶层 `reasoning_effort`——两种形态共存（K2 兼容栈读前者，K3 读后者）。显式给出 `thinking-effort` 即意味着开启思考，因此 `--thinking` 可以省略。K3 开源部署（`--think-mode opensource`）同理，旧版形态为 `chat_template_kwargs`。

| 模型类型 | 参数组合 | 发送的 `extra_body` |
|---------|---------|---------------------|
| Kimi 官方 K2系列 + 思考关闭| `--think-mode kimi` | `{"thinking": {"type": "disabled"}}` |
| Kimi 官方 K2系列 + 思考开启| `--thinking --think-mode kimi` | `{"thinking": {"type": "enabled"}}` |
| Kimi 官方 K3系列 + reasoning effort low | `--think-mode kimi --thinking-effort low` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "low"}, "reasoning_effort": "low"}` |
| Kimi 官方 K3系列 + reasoning effort high | `--think-mode kimi --thinking-effort high` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "high"}, "reasoning_effort": "high"}` |
| Kimi 官方 K3系列 + reasoning effort max | `--think-mode kimi --thinking-effort max` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "max"}, "reasoning_effort": "max"}` |
| 开源框架 K2系列 + 思考关闭 | `--think-mode opensource` | `{"chat_template_kwargs": {"thinking": false}}` |
| 开源框架 K2系列 + 思考开启 | `--thinking --think-mode opensource` | `{"chat_template_kwargs": {"thinking": true}}` |
| 开源框架 K3系列 + reasoning effort low | `--think-mode opensource --thinking-effort low` | `{"reasoning_effort": "low"}` |
| 开源框架 K3系列 + reasoning effort high | `--think-mode opensource --thinking-effort high` | `{"reasoning_effort": "high"}` |
| 开源框架 K3系列 + reasoning effort max | `--think-mode opensource --thinking-effort max` | `{"reasoning_effort": "max"}` |

### 查看结果

```bash
# 使用 inspect view 查看日志
uv run inspect view

# 日志保存在 logs/ 目录
```

`.eval` 日志中的每次模型调用都会包含一个 `request_context` 对象，其中
有 `client_request_id`、`request_id` 和上游服务返回的 `trace_id`。可以在
以下位置查看：

- `samples/<id>.json` → `metadata.request_context`（在 `inspect view` 的 **Metadata** 面板中显示）
- `samples/<id>.json` → `output.metadata.request_context`
- `samples/<id>.json` → `events[*].call.response._request_context`

### 恢复中断的评测

```bash
uv run inspect eval-retry logs/<log-file>.eval
```

## 注意事项

### AIME 2025

AIME 评测的输出 tokens 较多，需要注意：

1. 超时设置：客户端默认 `--client-timeout 86400`（24 小时），同时需要确认服务端、网关或代理的超时足够长。
2. 流式传输：强烈建议使用 `--stream`，非流式请求在 thinking 模式下更容易超时。
3. 并发控制：如果出现大量 429 或 `RemoteProtocolError`，降低 `--max-connections`。
4. 快速验证：建议先用 `--epochs 1` 跑通全部样本，确认配置正确后再运行完整评测。

```bash
# Step 1: 快速验证（30 samples x 1 epoch）
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream --epochs 1

# Step 2: 完整评测（30 samples x 32 epochs）
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --max-tokens 98304 --stream
```

### 自动重试机制

以下网络类错误会自动重试（指数退避，1-60 秒），无需手动配置：

| 错误类型 | 说明 |
|----------|------|
| `RateLimitError` / `429` | 服务端限流 |
| `APIConnectionError` | 连接失败 |
| `ReadError` / `RemoteProtocolError` | 网络读取错误 |

非网络类错误（如模型输出格式问题）不会重试，会直接记录到日志供后续分析。

## 项目结构

```text
├── eval.py              # 主评测入口 CLI
├── tests/params/        # 预检参数验证
├── tests/tool_call_json_schema/ # walle tool-call schema 校验
├── tests/toolcall_reject_cases/ # 畸形 tool-call 请求拒绝校验
├── tests/prompt_tokens/ # usage.prompt_tokens 准确性校验
├── tests/k3_features/   # K3 新功能 API contract 校验
├── kimi_model.py        # Kimi Model API 实现
├── aime2025.py          # AIME 2025 评测任务
├── mmmu_pro_vision.py   # MMMU Pro Vision 评测任务
├── ocr_bench.py         # OCRBench 评测任务
├── kimi_toolcall.py     # Kimi Tool Call 评测任务
├── testdata/            # JSON schema 与 prompt token 测试数据
├── toolcall_benchmark/  # Kimi Tool Call 测试数据，字段说明见目录内 README
├── beam/                # BEAM 1M context 数据和脚本，详情见目录内 README
├── logs/                # 评测日志
└── pyproject.toml       # 项目配置
```

## 联系我们

如果您有任何问题或建议，请联系 contact-kvv@kimi.com。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
