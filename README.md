# Kimi Vendor Verifier

基于 [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) 框架的模型评测工具，用于评测 Kimi 模型在各类 benchmark 上的表现。

## 支持的评测任务

| 任务 | 描述 | 数据集 |
|------|------|--------|
| **AIME 2025** | 美国数学邀请赛，评估数学推理能力 | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) |
| **MMMU Pro Vision** | 多模态理解评测（纯视觉） | [moonshotai/mmmu-pro-vision](https://huggingface.co/datasets/moonshotai/mmmu-pro-vision) (私有) |
| **OCRBench** | OCR 文字识别能力评测 | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) |

## 环境准备

### 1. 安装依赖

本项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync
```

或使用 pip：

```bash
pip install -e .
```

### 2. 配置环境变量

**方式一：通过 .env 文件配置（推荐）**

复制示例配置文件并填入你的配置：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件，填入你的 API Key 和 Base URL。

**方式二：通过 export 命令配置**

```bash
export KIMI_API_KEY="your-api-key"
export KIMI_BASE_URL="your-base-url"
```

### 3. Hugging Face 登录（MMMU Pro Vision 评测必需）

MMMU Pro Vision 使用私有数据集，运行评测前需要先登录 Hugging Face：

```bash
# 安装 huggingface-cli（如果尚未安装）
pip install huggingface_hub

# 登录 Hugging Face
huggingface-cli login
```

登录时需要提供有权限访问 `moonshotai/mmmu-pro-vision` 数据集的 Hugging Face Token。

## 运行评测

> 💡 **调试建议**：部署调试期间建议先使用 **OCRBench** 进行 debug，该数据集较小、运行速度快，适合快速验证部署是否正常。调试完成后再运行 MMMU Pro Vision 和 AIME 2025 正式评测。

### AIME 2025 评测

AIME 评测采用 32 次采样计算平均得分：

修改 `eval.py` 中的配置后运行：

```bash
uv run python eval.py
```

**参数说明：**
- `max_tokens=100000` - 最大输出 token 数
- `epochs=32` - 采样次数

### MMMU Pro Vision 评测

> ⚠️ **注意**：运行此评测前请确保已完成 [Hugging Face 登录](#3-hugging-face-登录mmmu-pro-vision-评测必需)

修改 `eval.py` 中的配置后运行：

```bash
uv run python eval.py
```

### OCRBench 评测

修改 `eval.py` 中的配置后运行 OCR 评测：

```bash
uv run python eval.py
```

## 查看评测结果

评测日志保存在 `logs/` 目录下，可使用 `inspect view` 查看：

```bash
uv run python inspect view
```

## 项目结构

```
├── aime2025.py          # AIME 2025 评测任务定义
├── mmmu_pro_vision.py   # MMMU Pro Vision 评测任务定义
├── ocr_bench.py         # OCRBench 评测任务定义
├── eval.py              # 主评测入口
├── logs/                # 评测日志目录
└── pyproject.toml       # 项目依赖配置
```

## 自定义评测

### 修改模型 ID

在 `eval.py` 中替换模型标识：

```python
eval(
    [aime2025, mmmu_pro_v, ocrbench],
    ["openai-api/kimi/{your-model-id}"],  # 替换为你的模型 ID
    ...
)
```

### 启用思考模式

在 `eval.py` 中可以通过 `extra_body` 启用模型思考模式：

```python
extra_body={
    "thinking": {"type": "enabled"},
}
```

### 关闭思考模式

在 `eval.py` 中可以通过 `extra_body` 关闭模型思考模式：

```python
extra_body={
    "thinking": {"type": "disabled"},
}
```

### vLLM / SGLang 思考模式

> ⚠️ **注意**：使用 vLLM 或 SGLang 部署时，思考模式的参数格式不同：

```python
extra_body={
    "chat_template_kwargs": {"thinking": True},   # 启用思考模式
}

extra_body={
    "chat_template_kwargs": {"thinking": False},  # 关闭思考模式
}
```

**Temperature 温度设置**
- **思考模式** `temperature=1.0`
- **非思考模式** `temperature=0.6`

### 调整并发和重试

```python
eval(
    ...
    max_connections=1000,  # 最大并发连接
    max_tasks=2,           # 最大并行任务数
    retry_on_error=3,      # 错误重试次数
    fail_on_error=True     # 遇错是否终止
)
```

## 评测指标

- **AIME 2025**: 使用 `accuracy` 和 `stderr` 指标，答案需符合 `Answer: <integer>` 或 `\boxed{<integer>}` 格式
- **MMMU Pro Vision**: 使用 `accuracy` 和 `stderr` 指标，答案需符合 `Answer: $LETTER` 格式
- **OCRBench**: 使用 `accuracy` 和 `stderr` 指标，通过答案子串匹配判断正确性
