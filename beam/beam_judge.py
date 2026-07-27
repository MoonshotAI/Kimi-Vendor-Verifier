#!/usr/bin/env python3
"""BEAM (Beyond a Million Tokens) LLM-as-Judge scoring.

Reads an answers.jsonl produced by beam_generate.py, judges every response
against its rubric items with an OpenAI-compatible judge model, writes
scores.jsonl and prints a summary (overall + per memory ability).

Scoring semantics replicate the official BEAM evaluation:
  - per rubric item, 3-level score (0.0 / 0.5 / 1.0), truncated with int()
    (0.5 -> 0), averaged across the question's rubric items
  - event_ordering is strict: 1.0 only if all events present in perfect order
  - empty responses score 0 (except abstention, where empty may be correct)

Resumable: questions already present in the scores file are skipped.

Example:
    export JUDGE_API_KEY=sk-...
    python beam_judge.py \
        --answers answers.jsonl \
        --judge-model gpt-oss-120b --judge-base-url https://api.example.com/v1 \
        --judge-reasoning-effort high --judge-max-tokens 16384 \
        --concurrency 32 --output scores.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("openai package is required: pip install openai")

QUESTION_TYPES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]

# ---------------------------------------------------------------------------
# Judge prompts (aligned with the official BEAM unified_llm_judge_base_prompt;
# identical to the prompts used in our lm-evaluation-harness runs)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator tasked with judging whether the LLM's response \
demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): {question}
- RUBRIC CRITERION (what to check): {rubric_item}
- RESPONSE TO EVALUATE: {llm_response}

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior \
that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often \
indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and \
attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive \
to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms \
(e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). \
Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** \
(extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly \
requires a format/structure (e.g., "itemized list", "no citations", \
"one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies \
the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed \
(allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element \
but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is \
non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for \
something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements \
connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the \
required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion \
according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied \
and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation]"
}}

NOTE: ONLY output the json object, without any explanation before or after that"""

# Event ordering: strict evaluation — all events present AND in perfect order → 1.0, else 0.0
EVENT_ORDERING_JUDGE_TEMPLATE = """\
You are an expert evaluator. The user asked the LLM to list events in the \
exact order they were discussed across a conversation.

## QUESTION
{question}

## CORRECT EVENT ORDER (reference)
{reference_order}

## MODEL'S RESPONSE
{llm_response}

## EVALUATION CRITERIA
The model's response is correct ONLY if **both** conditions are met:

1. **Complete coverage**: ALL reference events are mentioned in the response. \
Use semantic matching — paraphrases and synonyms are acceptable, but every \
reference event must have a clear corresponding item in the response.

2. **Perfect order**: The events appear in **exactly** the same relative \
order as the reference list. Any ordering error (swap, shift, reversal) \
means failure.

**Scoring**:
- **1.0**: ALL reference events are present AND in the exact correct order.
- **0.0**: Any event is missing OR any ordering error exists.

Be strict: partial coverage or almost-correct order still scores 0.0.

## OUTPUT FORMAT
Return a JSON object:
- "all_present": true/false (are ALL reference events mentioned?)
- "order_correct": true/false (are mentioned events in perfect order?)
- "score": 1.0 or 0.0
- "explanation": brief reason

Example:
{{"all_present": true, "order_correct": true, "score": 1.0, "explanation": "All 5 events present in correct order"}}
{{"all_present": true, "order_correct": false, "score": 0.0, "explanation": "All events present but items 3 and 4 are swapped"}}
{{"all_present": false, "order_correct": true, "score": 0.0, "explanation": "Only 3/5 events mentioned"}}

Output ONLY the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Score parsing (identical to our lm-harness implementation)
# ---------------------------------------------------------------------------

def _parse_judge_score(response: str) -> int:
    """Parse the 3-level score; official code truncates with int(): 0.5 -> 0."""
    try:
        obj = json.loads(response)
        return int(float(obj.get("score", 0)))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    json_match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', response)
    if json_match:
        try:
            return int(float(json_match.group(1)))
        except ValueError:
            pass

    score_match = re.search(r'"?score"?\s*:\s*([\d.]+)', response)
    if score_match:
        try:
            return int(float(score_match.group(1)))
        except ValueError:
            pass

    return 0


def _parse_float_score(response: str) -> float:
    try:
        obj = json.loads(response)
        return float(obj.get("score", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = re.search(r'"score"\s*:\s*([\d.]+)', response)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0


# ---------------------------------------------------------------------------
# Judge model access
# ---------------------------------------------------------------------------

async def call_judge(client: AsyncOpenAI, args, prompt: str) -> str:
    """One judge call with retries. Returns the content string (may be empty)."""
    kwargs = {
        "model": args.judge_model,
        "messages": [{"role": "user", "content": prompt.strip()}],
        "temperature": args.judge_temperature,
        "max_tokens": args.judge_max_tokens,
    }
    if args.judge_reasoning_effort:
        kwargs["reasoning_effort"] = args.judge_reasoning_effort

    last_err = None
    for attempt in range(args.max_retries):
        try:
            completion = await client.chat.completions.create(
                **kwargs, timeout=args.timeout,
            )
            if not completion.choices:
                return ""
            content = completion.choices[0].message.content
            # Reasoning judges occasionally exhaust max_tokens on thinking and
            # return content=None — treat as unparseable (score 0), never crash.
            return content if isinstance(content, str) else ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"  judge attempt {attempt + 1} failed: {e!r}; retry in {wait}s",
                  flush=True)
            await asyncio.sleep(wait)
    raise RuntimeError(f"judge call failed after {args.max_retries} retries: {last_err!r}")


async def judge_rubric_item(client, args, question: str, rubric_item: str,
                            llm_response: str) -> int:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, rubric_item=rubric_item, llm_response=llm_response,
    )
    try:
        judge_response = await call_judge(client, args, prompt)
    except Exception:
        return 0
    if not judge_response:
        return 0
    return _parse_judge_score(judge_response)


async def judge_event_ordering(client, args, question: str, rubric_items: list,
                               llm_response: str) -> float:
    reference_order = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(rubric_items))
    prompt = EVENT_ORDERING_JUDGE_TEMPLATE.format(
        question=question, reference_order=reference_order, llm_response=llm_response,
    )
    try:
        judge_response = await call_judge(client, args, prompt)
    except Exception:
        return 0.0
    if not judge_response:
        return 0.0
    score = _parse_float_score(judge_response)
    return 1.0 if score >= 1.0 else 0.0


async def score_row(client, args, row: dict) -> float:
    prediction = row.get("response") or ""
    question_type = row["question_type"]
    question = row["question"]

    # Empty-response guard: non-abstention empty responses are always wrong.
    if not prediction.strip() and question_type != "abstention":
        return 0.0

    rubric_items = row.get("rubric") or []
    if isinstance(rubric_items, str):
        try:
            rubric_items = json.loads(rubric_items)
        except json.JSONDecodeError:
            rubric_items = [rubric_items]
    if not rubric_items:
        rubric_items = ["The response should be relevant and correct."]

    if question_type == "event_ordering":
        return await judge_event_ordering(client, args, question, rubric_items, prediction)

    item_scores = await asyncio.gather(*(
        judge_rubric_item(client, args, question, item, prediction)
        for item in rubric_items
    ))
    return sum(item_scores) / len(item_scores)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def row_key(row: dict) -> str:
    return f"{row['chat_id']}|{row['question_type']}|{row['question_index']}"


def load_scores(path: str) -> dict:
    scores = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scores[f"{r['chat_id']}|{r['question_type']}|{r['question_index']}"] = r["score"]
    return scores


def print_summary(rows: list, scores: dict) -> None:
    per_type = {qt: [] for qt in QUESTION_TYPES}
    all_scores = []
    for row in rows:
        s = scores.get(row_key(row))
        if s is None:
            continue
        all_scores.append(s)
        if row["question_type"] in per_type:
            per_type[row["question_type"]].append(s)

    print(f"\n=== BEAM summary (n={len(all_scores)}) ===")
    if all_scores:
        print(f"overall: {sum(all_scores) / len(all_scores):.4f}")
    for qt in QUESTION_TYPES:
        vals = per_type[qt]
        if vals:
            print(f"  {qt}: {sum(vals) / len(vals):.4f} (n={len(vals)})")


async def run(args) -> None:
    rows = []
    with open(args.answers, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"loaded {len(rows)} answers from {args.answers}", flush=True)

    scores = load_scores(args.output)
    if scores:
        print(f"reused {len(scores)} existing scores from {args.output}", flush=True)
    todo = [r for r in rows if row_key(r) not in scores]
    if args.limit:
        todo = todo[: args.limit]
    print(f"to judge: {len(todo)}", flush=True)

    if todo:
        client = AsyncOpenAI(
            base_url=args.judge_base_url.rstrip("/"),
            api_key=args.judge_api_key,
            max_retries=0,
        )
        sem = asyncio.Semaphore(args.concurrency)
        out = open(args.output, "a", encoding="utf-8")
        t0 = time.time()
        counter = {"n": 0}

        async def worker(row):
            async with sem:
                score = await score_row(client, args, row)
            key = row_key(row)
            scores[key] = score
            out.write(json.dumps({
                "chat_id": row["chat_id"],
                "question_type": row["question_type"],
                "question_index": row["question_index"],
                "score": score,
            }, ensure_ascii=False) + "\n")
            out.flush()
            counter["n"] += 1
            n = counter["n"]
            if n % 20 == 0 or n == len(todo):
                rate = n / (time.time() - t0) * 60
                print(f"judged {n}/{len(todo)} ({rate:.1f} q/min)", flush=True)

        await asyncio.gather(*(worker(r) for r in todo))
        out.close()

    print_summary(rows, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True, help="answers.jsonl from beam_generate.py")
    parser.add_argument("--output", default="scores.jsonl", help="scores JSONL path")
    parser.add_argument("--judge-model", required=True, help="judge model name")
    parser.add_argument("--judge-base-url", required=True, help="judge endpoint base URL")
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", ""),
                        help="judge API key (default: env JUDGE_API_KEY)")
    parser.add_argument("--judge-reasoning-effort", default="",
                        help="optional reasoning_effort for reasoning judges (e.g. high)")
    parser.add_argument("--judge-max-tokens", type=int, default=16384)
    parser.add_argument("--judge-temperature", type=float, default=0.3)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--limit", type=int, default=0, help="only judge first N rows")
    args = parser.parse_args()

    if not args.judge_api_key:
        sys.exit("missing judge API key: pass --judge-api-key or export JUDGE_API_KEY")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
