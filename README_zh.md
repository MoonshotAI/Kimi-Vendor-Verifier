# [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)

[English](README.md) | 中文

## K3 评测结果
按照提交测试的时间顺序排列

model: kimi-k3

thinking effort: max

| Provider | OCRBench | MMMU Pro Vision | BEAM (1M) | DeepSWE |
|----------|----------|-----------------|-----------|---------|
| Moonshot | 0.89 | 0.82 | 0.31 | 0.675 |
| Fireworks  | 0.89 | 0.82 | 0.3037 | 0.664 |
| Baseten | 0.889 | 0.804 | 0.2975 | TBD |
| Together | 0.892 | 0.817 | TBD | TBD |
| DigitalOcean | 0.89 | 0.816 | TBD | TBD |
| Inferact | 0.893 | TBD | TBD | TBD |
| Nebius | 0.878 | 0.814 | TBD | TBD |
| Modal | 0.887 | 0.817 | 0.322 | 0.658 |




## 概览

| 类型 | 任务 | 说明 |
|------|------|------|
| Inspect-ai benchmark | OCRBench | OCR 文字识别能力评测 |
| Inspect-ai benchmark | MMMU Pro Vision | 多模态理解评测（视觉问答） |
| Inspect-ai benchmark | AIME 2025 | 数学推理能力评测；K3 不再使用 |
| 独立脚本 | BEAM (1M) | 1M token 长对话长期记忆评测，见 [beam/](beam/README_zh.md) |
| Pytest verifier | `tests/params/` | API 参数约束预检 |
| Pytest verifier | `tests/tool_call_json_schema/` | walle-valid MFJS schema 的 tool-call 参数校验 |
| Pytest verifier | `tests/k3_features/` | K3 feature contract 校验，包括 dynamic tools、response_format、tool_choice、thinking effort |
| Pytest verifier | `tests/prompt_tokens/` | 验证 vendor 上报的 `usage.prompt_tokens` 是否与期望常量一致 |
| Agent benchmark | [DeepSWE](https://github.com/datacurve-ai/deep-swe) | 多步 tool 使用和 coding-agent 能力评测，使用 Pier 平台评测 |


## 环境准备

### 安装依赖

```bash
uv sync && uv pip install -e .
```

### 配置环境变量

```bash
export KIMI_API_KEY="your-api-key"
export KIMI_BASE_URL="your-base-url"
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

验证 vendor 能否把 walle-valid MFJS schema 作为 tool-call `parameters` 使用，并返回符合 schema 的 `tool_calls[].function.arguments`。

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

#### 参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--smoke-model` | vendor 模型名称 | 环境变量 `MODEL_NAME` |
| `--base-url` | API base URL | 环境变量 `KIMI_BASE_URL`，否则 Moonshot API |
| `--api-key` | API key | 环境变量 `KIMI_API_KEY` |
| `--case-dir` | walle `validator_cases` 目录 | `testdata/walle_validator_cases/validator_cases` |
| `--selection` | case 选择模式：`all` / `explicit` / `object` | `all` |
| `--max-cases` | 最多运行多少个已选 case | 不限制 |
| `--thinking` | 开启 tool-call 请求的 thinking mode | 关闭 |
| `--think-mode` | thinking 参数格式：`kimi`、`opensource` 或 `none` | 环境变量 `THINK_MODE`，否则 `kimi` |
| `--max-tokens` | 每次 tool-call response 的最大输出 token 数 | `2048` |
| `--tool-json-report` | 额外的 JSON 报告产物路径 | `tool-call-schema-report.json` |

### K3 Feature 验证：`tests/k3_features/`

验证 K3 新功能 API，例如 dynamic tools、`response_format`、`tool_choice` 和 thinking effort。

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

CI job `verify_k3_features` 会运行 pytest 套件并从测试日志中报告结果。

### Prompt Token 验证：`tests/prompt_tokens/`

验证 vendor 上报的 `usage.prompt_tokens` 是否准确。该检查把 `testdata/prompt_token_cases/cases.jsonl` 中的每条 case 以 `stream=true` 发送，并将流中上报的 `usage.prompt_tokens` 与 case 中存储的期望常量做精确比对。case 格式见 [testdata/prompt_token_cases/README.md](testdata/prompt_token_cases/README.md)。

```bash
uv run pytest tests/prompt_tokens \
    --base-url "${KIMI_BASE_URL}" \
    --api-key "${KIMI_API_KEY}" \
    --smoke-model "${MODEL_NAME}" \
    -ra -v
```


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

| Benchmark | 模式（thinking, preserve thinking, effort） | Temperature | TopP | Max Tokens | Epochs |
|-----------|---------------------------------------------|-------------|------|------------|--------|
| OCRBench | Thinking, Keep all, low | 1.0 | 0.95 | 16384 | 1 |
| OCRBench | Thinking, Keep all, high | 1.0 | 0.95 | 16384 | 1 |
| OCRBench | Thinking, Keep all, max | 1.0 | 0.95 | 16384 | 1 |
| MMMU | Thinking, Keep all, low | 1.0 | 0.95 | 98304 | 1 |
| MMMU | Thinking, Keep all, high | 1.0 | 0.95 | 98304 | 1 |
| MMMU | Thinking, Keep all, max | 1.0 | 0.95 | 98304 | 1 |

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
    --max-tokens 16384 \
    --thinking \
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
    --max-tokens 98304 \
    --thinking \
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

## Agent Benchmark（DeepSWE）

为覆盖这类能力，我们额外增加了基于开源 [DeepSWE](https://github.com/datacurve-ai/deep-swe) 的 agent benchmark。DeepSWE 共 113 道 coding-agent 题目，这里使用开源 Pier 平台运行。agent runtime 使用 Kimi-code `0.23.6`及以上版本。

使用 Pier 平台运行 benchmark 前，需要先将 `kimi-code` 注册到 Pier agent 上。Kimi-code `0.23.6` 的配置参考：https://github.com/MoonshotAI/kimi-code/tree/%40moonshot-ai/kimi-code%400.23.6

## Inspect-ai 参数说明

### 可用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `benchmark` | 评测任务：`ocrbench`、`mmmu`、`aime2025` | `ocrbench` |
| `--model` | 模型标识，例如 `kimi/your-model-id` | **必填** |
| `--max-tokens` | 最大输出 token 数，见推荐参数配置 | **必填** |
| `--thinking` | 开启思考模式 | 关闭 |
| `--think-mode` | 思考参数格式：`none`、`kimi` 或 `opensource` | `none` |
| `--temperature` | 采样温度 | 可不设置，由服务端或模型默认值决定 |
| `--top-p` | Top-p 采样 | 可不设置，由服务端或模型默认值决定 |
| `--stream` | 启用流式传输，推荐用于长推理请求 | 关闭 |
| `--max-connections` | 最大并发连接数 | 按 benchmark |
| `--epochs` | 采样次数 | 按 benchmark |
| `--client-timeout` | HTTP 超时时间，单位秒 | `86400` |
| `--thinking-effort` | K3 reasoning effort 参数：`low`、`high`、`max` | 无 |

### 思考模式参数

| 模型类型 | 参数组合 | 发送的 `extra_body` |
|---------|---------|---------------------|
| Kimi 官方 + 思考关闭 | `--think-mode kimi` | `{"thinking": {"type": "disabled"}}` |
| Kimi 官方 + 思考开启 | `--thinking --think-mode kimi` | `{"thinking": {"type": "enabled"}}` |
| Kimi 官方 + reasoning effort low | `--thinking --think-mode kimi --thinking-effort low` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "low"}}` |
| Kimi 官方 + reasoning effort high | `--thinking --think-mode kimi --thinking-effort high` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "high"}}` |
| Kimi 官方 + reasoning effort max | `--thinking --think-mode kimi --thinking-effort max` | `{"thinking": {"type": "enabled", "keep": "all", "effort": "max"}}` |
| 开源框架 + 思考关闭 | `--think-mode opensource` | `{"chat_template_kwargs": {"thinking": false}}` |
| 开源框架 + 思考开启 | `--thinking --think-mode opensource` | `{"chat_template_kwargs": {"thinking": true}}` |
| 开源框架 + reasoning effort low | `--thinking --think-mode opensource --thinking-effort low` | `{"chat_template_kwargs": {"thinking": true, "preserve_thinking": true, "thinking_effort": "low"}}` |
| 开源框架 + reasoning effort high | `--thinking --think-mode opensource --thinking-effort high` | `{"chat_template_kwargs": {"thinking": true, "preserve_thinking": true, "thinking_effort": "high"}}` |
| 开源框架 + reasoning effort max | `--thinking --think-mode opensource --thinking-effort max` | `{"chat_template_kwargs": {"thinking": true, "preserve_thinking": true, "thinking_effort": "max"}}` |


### 查看结果

```bash
# 使用 inspect view 查看日志
uv run inspect view

# 日志保存在 logs/ 目录
```

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
├── tests/prompt_tokens/ # usage.prompt_tokens 准确性校验
├── tests/k3_features/   # K3 新功能 API contract 校验
├── kimi_model.py        # Kimi Model API 实现
├── aime2025.py          # AIME 2025 评测任务
├── mmmu_pro_vision.py   # MMMU Pro Vision 评测任务
├── ocr_bench.py         # OCRBench 评测任务
├── testdata/            # JSON schema 与 prompt token 测试数据
├── beam/                # BEAM 1M context 数据和脚本，详情见目录内 README
├── logs/                # 评测日志
└── pyproject.toml       # 项目配置
```

## 联系我们

如果您有任何问题或建议，请联系 contact-kvv@kimi.com。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
