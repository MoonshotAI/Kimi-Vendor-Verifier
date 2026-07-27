# BEAM（Beyond a Million Tokens）— 1M 长期记忆基准测试

BEAM 通过在最长可达 1M token 的多轮对话上提出 probing questions，评估大语言模型的长期记忆能力。该基准覆盖 10 类记忆能力：拒答、矛盾消解、事件排序、信息抽取、指令遵循、知识更新、多会话推理、偏好遵循、摘要、时间推理。

- 论文：https://arxiv.org/abs/2510.27246
- 官方仓库：https://github.com/mohammadtavakoli78/BEAM
- 数据集：35 段对话 x 每段 20 个 probing questions = 700 个问题（1M 规模，随 `data/` 提供，由 `prepare_data.py` 从官方发布版本转换而来）

与本仓库中的其他 benchmark 不同，BEAM 以两个独立脚本运行。由于 1M 上下文的答案生成通常需要数小时，答案生成和判分被拆分成两个阶段，并且两者都支持断点续跑：

1. `beam_generate.py`：将完整对话历史 + probing question 发送给被测模型，逐行写入 `answers.jsonl`。
2. `beam_judge.py`：使用 LLM-as-Judge 按 rubric 为每个答案打分，写入 `scores.jsonl` 并打印汇总结果。

## 环境准备

```bash
pip install openai transformers   # transformers 仅在需要截断时使用
```

数据已随目录提供（`data/beam_1m_chats.jsonl.gz` + `data/beam_1m_questions.jsonl`）。如果需要从官方 BEAM checkout 重新生成：

```bash
python prepare_data.py --src /path/to/BEAM/chats/1M --tag 1m
```

## 1. 生成答案

```bash
export BEAM_API_KEY="your-api-key"
python beam_generate.py \
    --model your-model-id \
    --base-url https://your-endpoint/v1 \
    --concurrency 16 \
    --output answers.jsonl
```

Thinking mode 示例（嵌套 body 会原样透传）：

```bash
python beam_generate.py \
    --model your-model-id \
    --base-url https://your-endpoint/v1 \
    --thinking-json '{"thinking":{"type":"enabled","keep":"all","effort":"max"}}' \
    --output answers.jsonl
```

### 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填） | endpoint 提供的模型名称 |
| `--base-url` | （必填） | OpenAI-compatible base URL |
| `--api-key` | 环境变量 `BEAM_API_KEY` | API key |
| `--temperature` | 1.0 | 采样 temperature |
| `--top-p` | 0.95 | nucleus sampling 参数 |
| `--max-tokens` | 32768 | 最大输出 token 数 |
| `--thinking-json` | （空） | 额外请求 body JSON，例如 thinking 配置 |
| `--tokenizer` | bundled Kimi-K2.6 | 用于截断的 HF id 或本地 tokenizer 路径；为空表示不截断 |
| `--max-context-tokens` | 1048576 | 内容预算 = max-context - max-tokens - 5000 |
| `--concurrency` | 16 | 并发请求数 |
| `--max-retries` | 8 | 单个问题在记录空答案前的最大重试次数 |
| `--limit` | 0（全部） | 只运行前 N 个问题，用于 smoke test |
| `--dry-run` | 关闭 | 只构造 prompts 并打印统计信息，不发送请求 |

说明：

- Prompt 是扁平化后的完整对话历史（无 system prompt），末尾追加 probing question，并使用官方前缀 `NOTE: Only provide the answer without any explanations.`。
- 当内容超过 token 预算时，截断会从开头删除完整 message，并尽量保持 user/assistant 成对结构。默认使用随仓库提供的 Kimi-K2.6 tokenizer（`tokenizer/`，从官方模型发布版本中提取）；也可以通过 `--tokenizer` 指向其他 HF id 或本地路径，以匹配不同模型系列。
- 脚本重启时会跳过输出文件中已经存在的问题。
- 如果某个问题所有重试都失败，脚本会写入一个空 response 和 `error` 字段，而不是中止整个运行；该问题会得 0 分。

## 2. 判分答案

```bash
export JUDGE_API_KEY="your-judge-key"
python beam_judge.py \
    --answers answers.jsonl \
    --judge-model your-judge-model \
    --judge-base-url https://your-judge-endpoint/v1 \
    --output scores.jsonl
```

Reasoning judge 示例：

```bash
python beam_judge.py \
    --answers answers.jsonl \
    --judge-model gpt-oss-120b \
    --judge-base-url https://your-judge-endpoint/v1 \
    --judge-reasoning-effort high --judge-max-tokens 16384 \
    --concurrency 32 \
    --output scores.jsonl
```

### 判分参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--answers` | （必填） | `beam_generate.py` 生成的 answers.jsonl |
| `--judge-model` | （必填） | judge 模型名称 |
| `--judge-base-url` | （必填） | judge endpoint base URL |
| `--judge-api-key` | 环境变量 `JUDGE_API_KEY` | judge API key |
| `--judge-reasoning-effort` | （空） | reasoning judge 可选的 `reasoning_effort` 参数 |
| `--judge-max-tokens` | 16384 | judge 模型最大输出 token 数 |
| `--judge-temperature` | 0.3 | judge 模型 temperature |
| `--concurrency` | 32 | 并发 judge 调用数 |

> **关于 `--judge-temperature` 的说明**：不同 judge 模型对允许的温度值限制可能不同。例如 `kimi-k2.6` 要求 `temperature=1.0`，此时需要额外传入 `--judge-temperature 1.0`。

## 评分语义（与官方 BEAM evaluation 对齐）

- 每个问题都带有 rubric（criteria 列表）。judge 会对每条 criterion 按 3 档分数打分（0.0 / 0.5 / 1.0）；分数会用 `int()` 截断（0.5 -> 0），再对所有 criteria 求平均。这复刻了官方实现。
- `event_ordering` 采用严格评分：只有所有参考事件都出现且顺序完全正确时才得 1.0，否则得 0.0。
- 空回答直接得 0，不调用 judge（`abstention` 除外，因为空回答或拒答可能是正确的）。
- 无法解析或调用失败的 judge response 记 0 分，永远不会中止整轮运行。
- 最终报告包含 overall mean 和各能力维度 mean，与官方 per-ability breakdown 对齐。

## 输出格式

`answers.jsonl`：每个问题一行：
`{chat_id, question_type, question_index, question, gold_answer, rubric, response, finish_reason, prompt_tokens, completion_tokens, error}`

`scores.jsonl`：每个问题一行：
`{chat_id, question_type, question_index, score}`
