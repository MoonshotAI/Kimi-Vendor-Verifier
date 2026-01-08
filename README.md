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
# 运行单个 benchmark
uv run python eval.py <benchmark> --model kimi/<model-id> [options]

# 运行所有 benchmark
uv run python eval.py all --model kimi/<model-id>
```

### 可用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `benchmark` | 评测任务: `aime2025`, `mmmu`, `ocrbench`, `all` | `all` |
| `--model` | 模型标识，如 `kimi/your-model-id` | **必填** |
| `--thinking` | 启用思考模式 | 关闭 |
| `--think-mode` | 思考模式格式: `kimi`, `vllm`, 或 `none` | `kimi` |
| `--stream` | 启用流式传输（推荐，避免长推理超时） | 关闭 |
| `--retry` | 错误重试次数 | `0` |
| `--max-connections` | 最大并发连接数 | 按 benchmark 配置 |
| `--epochs` | 采样次数（仅 AIME） | 按 benchmark 配置 |
| `--client-timeout` | HTTP 超时时间（秒） | `86400` |

> **`--think-mode` 说明**：
> - `kimi`: 混合模型，使用 `{"thinking": {"type": "enabled/disabled"}}`
> - `vllm`: vLLM/SGLang 部署，使用 `{"chat_template_kwargs": {"thinking": true/false}}`
> - `none`: 非混合模型，不传递 thinking 参数

### 示例

```bash
# AIME 2025 评测（混合模型 + 思考模式 + 流式）
uv run python eval.py aime2025 --model kimi/your-model-id --thinking --stream

# OCRBench 评测（快速验证部署）
uv run python eval.py ocrbench --model kimi/your-model-id --stream

# MMMU Pro Vision 评测
uv run python eval.py mmmu --model kimi/your-model-id --thinking

# 非混合模型（不传递 thinking 参数）
uv run python eval.py aime2025 --model kimi/your-model-id --think-mode none --stream

# vLLM/SGLang 部署的模型
uv run python eval.py aime2025 --model kimi/your-model-id --thinking --think-mode vllm
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

## 默认配置

| Benchmark | Temperature | Max Tokens | Epochs |
|-----------|-------------|------------|--------|
| OCRBench | 0.6 / 1.0 | 8K / 16K | 1 |
| MMMU | 0.6 / 1.0 | 32K / 64K | 1 |
| AIME 2025 | 0.6 / 1.0 | 32K / 96K | 32 |

> 左侧为非思考模式，右侧为思考模式配置

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
uv run python eval.py aime2025 --model kimi/your-model-id --thinking --stream --epochs 1

# Step 2: 完整评测（30 samples x 32 epochs）
uv run python eval.py aime2025 --model kimi/your-model-id --thinking --stream
```

### 自动重试机制

以下情况会自动重试（无限重试直到成功）：

| 错误类型 | 说明 |
|----------|------|
| `RateLimitError` / `429` | 服务端限流，等待后重试 |
| `APIConnectionError` | 连接建立失败 |
| `ReadError` / `RemoteProtocolError` | 网络读取错误（仅在未开始接收数据时重试） |

**不会重试的情况**：
- 流传输已开始接收数据后中断（避免重复请求导致结果不一致）
- 其他非网络类错误（如参数错误、认证失败等）

### 流传输中断处理

当流传输中断时（如 `RemoteProtocolError`）：
- **已开始接收数据**: 返回已收到的部分内容，**不重试**（服务端已处理，重试会导致重复计算）
- **未开始接收数据**: **自动重试**（连接失败，安全重试）

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `RateLimitError: 429` | 并发过高触发限流 | 降低 `--max-connections` |
| `RemoteProtocolError` | 服务端异常断开连接 | 降低并发，检查服务端日志 |
| `Request timed out` | 客户端超时 | 增加 `--client-timeout` |
| `finish_reason: length` | 输出超过 max_tokens | 增加 max_tokens 配置 |
