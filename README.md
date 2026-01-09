# Kimi Vendor Verifier

基于 [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) 框架的模型评测工具，用于评测 Kimi 模型在各类 benchmark 上的表现。

## 支持的评测任务

| 任务 | 描述 | 数据集 |
|------|------|--------|
| **AIME 2025** | 美国数学邀请赛，评估数学推理能力 | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) |
| **MMMU Pro Vision** | 多模态理解评测（纯视觉） | [MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| **OCRBench** | OCR 文字识别能力评测 | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) |

## 环境准备

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv sync && uv pip install -e .

# 或使用 pip
pip install -e .
```

### 2. 配置环境变量

```bash
export KIMI_API_KEY="your-api-key"
export KIMI_BASE_URL="your-base-url"
```

或复制 `.env.example` 到 `.env` 并填入配置。

## 运行评测

### 基本用法

```bash
# 运行单个 benchmark（必填参数：--model, --temperature, --max-tokens）
uv run python eval.py <benchmark> --model kimi/<model-id> \
    --temperature <temp> --max-tokens <tokens> [options]

# 示例：运行 AIME 2025
uv run python eval.py aime2025 --model kimi/your-model-id \
    --temperature 0.6 --max-tokens 32768 --stream
```

### 可用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `benchmark` | 评测任务: `aime2025`, `mmmu`, `ocrbench`, `all` | `all` |
| `--model` | 模型标识，如 `kimi/your-model-id` | **必填** |
| `--temperature` | 采样温度（思考 1.0，非思考 0.6） | **必填** |
| `--max-tokens` | 最大输出 token 数（见下方推荐配置表） | **必填** |
| `--thinking` | 开启思考模式（需配合 `--think-mode kimi/vllm`） | 关闭 |
| `--think-mode` | 思考参数格式：`none`（非混合模型）、`kimi`、`vllm` | `none` |
| `--stream` | 启用流式传输（推荐，避免长推理超时） | 关闭 |
| `--max-connections` | 最大并发连接数 | 按 benchmark |
| `--epochs` | 采样次数 | 按 benchmark |
| `--client-timeout` | HTTP 超时时间（秒） | `86400` |

> **思考模式参数说明**：
>
> | 模型类型 | 参数组合 | 发送的 extra_body |
> |---------|---------|------------------|
> | 非混合模型 | 不传 `--think-mode` | `{}` |
> | 混合模型 + 思考关闭 | `--think-mode kimi` | `{"thinking": {"type": "disabled"}}` |
> | 混合模型 + 思考开启 | `--thinking --think-mode kimi` | `{"thinking": {"type": "enabled"}}` |
> | vLLM 部署 + 思考关闭 | `--think-mode vllm` | `{"chat_template_kwargs": {"thinking": false}}` |
> | vLLM 部署 + 思考开启 | `--thinking --think-mode vllm` | `{"chat_template_kwargs": {"thinking": true}}` |

### 示例

```bash
# 非混合模型（不传 --think-mode）
uv run python eval.py aime2025 --model kimi/your-model-id \
    --temperature 0.6 --max-tokens 32768 --stream

# 混合模型 + 思考开启 (Kimi API)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --temperature 1.0 --max-tokens 98304 --stream

# 混合模型 + 思考关闭 (Kimi API)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --think-mode kimi --temperature 0.6 --max-tokens 32768 --stream

# 混合模型 + 思考开启 (vLLM/SGLang)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode vllm --temperature 1.0 --max-tokens 98304 --stream
```

## 查看结果

```bash
# 使用 inspect view 查看日志
uv run inspect view

# 日志保存在 logs/ 目录
```

## 恢复中断的评测

```bash
uv run inspect eval-retry logs/<log-file>.eval
```

## 项目结构

```
├── eval.py              # 主评测入口 CLI
├── kimi_model.py        # Kimi Model API 实现
├── aime2025.py          # AIME 2025 评测任务
├── mmmu_pro_vision.py   # MMMU Pro Vision 评测任务
├── ocr_bench.py         # OCRBench 评测任务
├── logs/                # 评测日志
└── pyproject.toml       # 项目配置
```

## 推荐配置

| Benchmark | Temperature | Max Tokens | Epochs |
|-----------|-------------|------------|--------|
| OCRBench | 0.6 / 1.0 | 8192 / 16384 | 1 |
| MMMU | 0.6 / 1.0 | 32768 / 65536 | 1 |
| AIME 2025 | 0.6 / 1.0 | 32768 / 98304 | 32 |

> 格式：非思考模式 / 思考模式

## 评测指标

- **AIME 2025**: `accuracy` / `stderr`，答案格式 `Answer: <integer>` 或 `\boxed{<integer>}`
- **MMMU Pro Vision**: `accuracy` / `stderr`，答案格式 `Answer: $LETTER`
- **OCRBench**: `accuracy` / `stderr`，子串匹配判断

## 注意事项

### AIME 2025 评测

AIME 评测的输出 tokens 较多，需要注意：

1. **超时设置**
   - **客户端**: 默认 `--client-timeout 86400`（24小时），一般无需修改
   - **服务端**: 确保服务端的请求超时也设置足够长
   - **网关/代理**: 如使用 nginx/ALB，需调整 `proxy_read_timeout` 等配置

2. **流式传输**
   - **强烈建议**使用 `--stream` 参数
   - 非流式请求在 thinking 模式下容易超时
   - 流式可保持连接活跃，避免中间网关超时

3. **并发控制**
   - 默认 `max_connections=100`，根据服务端承载能力调整
   - 如果出现大量 429 或 `RemoteProtocolError`，降低并发数

4. **快速验证**
   - 建议先用 `--epochs 1` 跑通全部样本，确认配置正确
   - 验证通过后再运行 `--epochs 32` 完整评测

```bash
# Step 1: 快速验证（30 samples x 1 epoch）
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --temperature 1.0 --max-tokens 98304 --stream --epochs 1

# Step 2: 完整评测（30 samples x 32 epochs）
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --temperature 1.0 --max-tokens 98304 --stream
```

### 自动重试机制

以下网络类错误会**自动重试**（指数退避，1-60 秒），无需手动配置：

| 错误类型 | 说明 |
|----------|------|
| `RateLimitError` / `429` | 服务端限流 |
| `APIConnectionError` | 连接失败 |
| `ReadError` / `RemoteProtocolError` | 网络读取错误 |

> 非网络类错误（如模型输出格式问题）不会重试，会直接记录到日志供后续分析。

### 流传输中断处理

当流传输中断时（如 `RemoteProtocolError`）：
- **未开始接收数据**: **自动重试**（连接失败，安全重试）
- **已开始接收数据**: 返回已收到的部分内容，**不重试**（服务端已处理，重试会导致重复计算）
