# BEAM (Beyond a Million Tokens) — 1M Long-Term Memory Benchmark

Evaluates long-term memory abilities of LLMs via probing questions over
multi-turn conversations of up to 1M tokens, across 10 memory abilities:
abstention, contradiction resolution, event ordering, information extraction,
instruction following, knowledge update, multi-session reasoning, preference
following, summarization, temporal reasoning.

- Paper: https://arxiv.org/abs/2510.27246
- Official repo: https://github.com/mohammadtavakoli78/BEAM
- Dataset: 35 conversations x 20 probing questions = 700 questions (1M size,
  shipped under `data/`, converted from the official release by `prepare_data.py`)

Unlike the other benchmarks in this repository, BEAM runs as two standalone
scripts (generation takes hours at 1M context, so answer generation and
judging are decoupled, and both are resumable):

1. `beam_generate.py` — sends full conversation history + probing question to
   the model under test, writes `answers.jsonl` line by line.
2. `beam_judge.py` — scores each answer against its rubric with an
   LLM-as-Judge, writes `scores.jsonl` and prints a summary.

## Setup

```bash
pip install openai transformers   # transformers only needed for truncation
```

Data is included (`data/beam_1m_chats.jsonl.gz` + `data/beam_1m_questions.jsonl`).
To regenerate from an official BEAM checkout instead:

```bash
python prepare_data.py --src /path/to/BEAM/chats/1M --tag 1m
```

## 1. Generate answers

```bash
export BEAM_API_KEY="your-api-key"
python beam_generate.py \
    --model your-model-id \
    --base-url https://your-endpoint/v1 \
    --concurrency 16 \
    --output answers.jsonl
```

Thinking-mode example (nested body is passed through as-is):

```bash
python beam_generate.py \
    --model your-model-id \
    --base-url https://your-endpoint/v1 \
    --thinking-json '{"thinking":{"type":"enabled","keep":"all","effort":"max"}}' \
    --output answers.jsonl
```

### Generation parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | (required) | model name served by the endpoint |
| `--base-url` | (required) | OpenAI-compatible base URL |
| `--api-key` | env `BEAM_API_KEY` | API key |
| `--temperature` | 1.0 | sampling temperature |
| `--top-p` | 0.95 | nucleus sampling |
| `--max-tokens` | 32768 | max output tokens |
| `--thinking-json` | (empty) | extra request body JSON, e.g. thinking config |
| `--tokenizer` | bundled Kimi-K2.6 | HF id / local path for truncation; empty = no truncation |
| `--max-context-tokens` | 1048576 | content budget = max-context − max-tokens − 5000 |
| `--concurrency` | 16 | parallel requests |
| `--max-retries` | 8 | per-question retries before recording empty answer |
| `--limit` | 0 (all) | only run the first N questions (smoke test) |
| `--dry-run` | off | build prompts without sending requests |

Notes:

- The prompt is the full flattened conversation (no system prompt) with the
  probing question appended, using the official prefix
  `NOTE: Only provide the answer without any explanations.`
- Truncation drops whole messages from the front (keeping user/assistant
  pairs intact) when the content exceeds the token budget. By default the
  bundled Kimi-K2.6 tokenizer (`tokenizer/`, extracted from the official
  model release) is used; point `--tokenizer` at another HF id / local path
  to match a different model family.
- Restarting the script skips questions already present in the output file.
- A question that fails all retries is written with an empty response and an
  `error` field (it scores 0) instead of aborting the run.

## 2. Judge answers

```bash
export JUDGE_API_KEY="your-judge-key"
python beam_judge.py \
    --answers answers.jsonl \
    --judge-model your-judge-model \
    --judge-base-url https://your-judge-endpoint/v1 \
    --output scores.jsonl
```

Reasoning judge example:

```bash
python beam_judge.py \
    --answers answers.jsonl \
    --judge-model gpt-oss-120b \
    --judge-base-url https://your-judge-endpoint/v1 \
    --judge-reasoning-effort high --judge-max-tokens 16384 \
    --concurrency 32 \
    --output scores.jsonl
```

### Judge parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--answers` | (required) | answers.jsonl from `beam_generate.py` |
| `--judge-model` | (required) | judge model name |
| `--judge-base-url` | (required) | judge endpoint base URL |
| `--judge-api-key` | env `JUDGE_API_KEY` | judge API key |
| `--judge-reasoning-effort` | (empty) | optional `reasoning_effort` for reasoning judges |
| `--judge-max-tokens` | 16384 | judge max output tokens |
| `--judge-temperature` | 0.3 | judge temperature |
| `--concurrency` | 32 | parallel judge calls |

> **Note on `--judge-temperature`**: Different judge models may enforce different allowed temperature values. For example, `kimi-k2.6` requires `temperature=1.0`; in that case run with `--judge-temperature 1.0`.

## Scoring semantics (aligned with the official BEAM evaluation)

- Each question carries a rubric (list of criteria). The judge scores every
  criterion on a 3-level scale (0.0 / 0.5 / 1.0); scores are truncated with
  `int()` (0.5 → 0) and averaged across criteria — replicating the official
  implementation.
- `event_ordering` is strict: 1.0 only if all reference events are present in
  perfect order, otherwise 0.0.
- Empty responses score 0 without calling the judge (except `abstention`,
  where an empty/refusal response may be correct).
- Unparseable or failed judge responses score 0 (they never abort the run).
- The final report is the overall mean plus per-ability means, matching the
  official per-ability breakdown.

## Output format

`answers.jsonl` — one line per question:
`{chat_id, question_type, question_index, question, gold_answer, rubric, response, finish_reason, prompt_tokens, completion_tokens, error}`

`scores.jsonl` — one line per question:
`{chat_id, question_type, question_index, score}`
