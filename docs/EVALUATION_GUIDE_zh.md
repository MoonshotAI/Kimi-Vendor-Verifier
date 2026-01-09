# Kimi 模型厂商评测标准

本文档描述 Kimi 模型部署验证的评测项目和达标标准。

## 评测项目

| 评测项目 | 评测内容 | 仓库地址 |
|----------|----------|----------|
| **K2-Vendor-Verifier** | ToolCall 精度 | [github.com/MoonshotAI/K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) |
| **Kimi-Vendor-Verifier** | Benchmark 能力 | [github.com/MoonshotAI/Kimi-Vendor-Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier) |

---

## 一、K2-Vendor-Verifier

### 评测内容

验证 Kimi K2 模型在 ToolCall 场景下的精度。

### 评测指标

| 指标 | 说明 | 达标标准 |
|------|------|----------|
| **tool_call_f1** | ToolCall 触发的 F1 值 | ≥ 80% |
| **schema_accuracy** | ToolCall JSON Schema 正确率 | ≥ 98% |

### 评测参数

| 模式 | Temperature | Max Tokens |
|------|-------------|------------|
| 思考模式 (K2-thinking) | 1.0 | 65536 |
| 非思考模式 (K2-instruct) | 0.6 | 65536 |

### 运行方式

使用 `samples.jsonl` 测试集运行评测，生成 `results.jsonl` 结果文件，与官方 API 结果对比计算 F1 值。

```bash
# 非混合模型
python tool_calls_eval.py samples.jsonl \
    --model <model-name> \
    --base-url <api-endpoint> \
    --api-key <api-key> \
    --output results.jsonl

# 混合模型（需传 extra-body 控制思考模式）
python tool_calls_eval.py samples.jsonl \
    --model <model-name> \
    --base-url <api-endpoint> \
    --api-key <api-key> \
    --extra-body '{"thinking": {"type": "enabled"}}' \
    --output results.jsonl
```

> **混合模型 extra-body 参数**：
>
> | 部署方式 | 思考模式 | 非思考模式 |
> |----------|----------|------------|
> | Kimi API | `{"thinking": {"type": "enabled"}}` | `{"thinking": {"type": "disabled"}}` |
> | vLLM / SGLang | `{"chat_template_kwargs": {"thinking": true}}` | `{"chat_template_kwargs": {"thinking": false}}` |

详见：[K2-Vendor-Verifier README](https://github.com/MoonshotAI/K2-Vendor-Verifier#readme)

---

## 二、Kimi-Vendor-Verifier

### 评测内容

验证 Kimi 模型在标准 Benchmark 上的能力表现。

### 评测任务

| Benchmark | 描述 | 数据集 |
|-----------|------|--------|
| **AIME 2025** | 美国数学邀请赛 | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) |
| **MMMU Pro Vision** | 多模态理解（纯视觉） | [MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| **OCRBench** | OCR 文字识别 | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) |

### 评测参数与达标标准

#### 思考模式 (--thinking --think-mode kimi)

| Benchmark | Temperature | Max Tokens | Epochs | 达标标准 |
|-----------|-------------|------------|--------|----------|
| AIME 2025 | 1.0 | 98304 | 32 | accuracy ≥ **TBD** |
| MMMU Pro | 1.0 | 65536 | 1 | accuracy ≥ **TBD** |
| OCRBench | 1.0 | 16384 | 1 | accuracy ≥ **TBD** |

#### 非思考模式 (--think-mode kimi)

| Benchmark | Temperature | Max Tokens | Epochs | 达标标准 |
|-----------|-------------|------------|--------|----------|
| AIME 2025 | 0.6 | 32768 | 32 | accuracy ≥ **TBD** |
| MMMU Pro | 0.6 | 32768 | 1 | accuracy ≥ **TBD** |
| OCRBench | 0.6 | 8192 | 1 | accuracy ≥ **TBD** |

> **TBD**: 待基于官方 API 评测结果确定

### 运行方式

```bash
# 示例：AIME 2025 思考模式
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --temperature 1.0 --max-tokens 98304 --stream
```

详见：[Kimi-Vendor-Verifier README](https://github.com/MoonshotAI/Kimi-Vendor-Verifier#readme)

---

## 三、评测汇总

| 评测项 | 指标 | 达标标准 |
|--------|------|----------|
| K2 ToolCall | tool_call_f1 | ≥ 80% |
| K2 ToolCall | schema_accuracy | ≥ 98% |
| AIME 2025 (思考) | accuracy | TBD |
| MMMU Pro (思考) | accuracy | TBD |
| OCRBench | accuracy | TBD |

---

## 四、部署建议


