#!/usr/bin/env python3
"""BEAM (Beyond a Million Tokens) answer generation.

Sends each probing question together with its full conversation history to an
OpenAI-compatible chat-completions endpoint and writes one JSONL line per
question. Resumable: already-answered (chat_id, question_type, question_index)
tuples in the output file are skipped on restart.

Paper: https://arxiv.org/abs/2510.27246
Repo:  https://github.com/mohammadtavakoli78/BEAM

Example:
    export BEAM_API_KEY=sk-...
    python beam_generate.py \
        --model my-model --base-url https://api.example.com/v1 \
        --concurrency 16 \
        --thinking-json '{"thinking":{"type":"enabled","keep":"all","effort":"max"}}' \
        --tokenizer /path/to/Kimi-K2.6-tokenizer \
        --output answers.jsonl
"""

import argparse
import asyncio
import gzip
import json
import os
import sys
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHATS_FILE = os.path.join(DATA_DIR, "beam_1m_chats.jsonl.gz")
QUESTIONS_FILE = os.path.join(DATA_DIR, "beam_1m_questions.jsonl")

# Bundled Kimi-K2.6 tokenizer (extracted from the official model release) —
# used for truncation by default, no external files needed.
BUILTIN_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer")

# Official BEAM probing-question prefix (no system prompt, flat history).
PROBING_TEMPLATE = "NOTE: Only provide the answer without any explanations. \nQuestion: {question}"

# Reserve budget: chat-template overhead + probing question tokens.
RESERVE_TOKENS = 5000


def load_dataset(chats_file: str, questions_file: str) -> list:
    """Join questions with their chat history. Returns a list of docs."""
    chats = {}
    with gzip.open(chats_file, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chats[row["chat_id"]] = row["messages"]
    docs = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["chat_id"] not in chats:
                continue
            docs.append({**q, "messages": chats[q["chat_id"]]})
    return docs


def truncate_messages(messages: list, max_content_tokens: int, tokenizer) -> list:
    """Drop messages from the front until the content fits the token budget.

    Removes whole messages, keeping the first remaining message a user turn
    (pairs stay intact). Mirrors the truncation used in our lm-harness runs.
    """
    msg_tokens = [len(tokenizer.encode(m["content"])) for m in messages]
    total = sum(msg_tokens)
    if total <= max_content_tokens:
        return messages

    start_idx = 0
    while start_idx < len(messages) and total > max_content_tokens:
        total -= msg_tokens[start_idx]
        start_idx += 1
    while start_idx < len(messages) and messages[start_idx]["role"] != "user":
        start_idx += 1

    truncated = messages[start_idx:]
    print(f"  truncated chat from {len(messages)} to {len(truncated)} messages "
          f"(budget {max_content_tokens})", flush=True)
    return truncated


def build_request_messages(doc: dict, tokenizer, max_content_tokens: int) -> list:
    messages = doc["messages"]
    if tokenizer is not None:
        messages = truncate_messages(messages, max_content_tokens, tokenizer)
    return messages + [{
        "role": "user",
        "content": PROBING_TEMPLATE.format(question=doc["question"]),
    }]


def doc_key(doc: dict) -> str:
    return f"{doc['chat_id']}|{doc['question_type']}|{doc['question_index']}"


def load_done_keys(output_path: str) -> set:
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(f"{row['chat_id']}|{row['question_type']}|{row['question_index']}")
    return done


async def generate_one(client, doc: dict, args, extra_body: dict) -> dict:
    messages = build_request_messages(doc, args._tokenizer, args._max_content_tokens)
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    row = {
        "chat_id": doc["chat_id"],
        "question_type": doc["question_type"],
        "question_index": doc["question_index"],
        "question": doc["question"],
        "gold_answer": doc["gold_answer"],
        "rubric": doc["rubric"],
        "response": "",
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
    }

    last_err = None
    for attempt in range(args.max_retries):
        try:
            completion = await client.chat.completions.create(
                **payload, extra_body=extra_body or None, timeout=args.timeout,
            )
            choice = completion.choices[0] if completion.choices else None
            row["response"] = (choice.message.content or "") if choice else ""
            row["finish_reason"] = choice.finish_reason if choice else None
            if completion.usage:
                row["prompt_tokens"] = completion.usage.prompt_tokens
                row["completion_tokens"] = completion.usage.completion_tokens
            return row
        except Exception as e:  # noqa: BLE001 - retry on any API failure
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"  [{doc_key(doc)}] attempt {attempt + 1} failed: {e!r}; "
                  f"retry in {wait}s", flush=True)
            await asyncio.sleep(wait)

    # Give up: record an empty answer (scores 0) instead of aborting the run.
    row["error"] = repr(last_err)
    return row


async def run(args) -> None:
    docs = load_dataset(args.chats_file, args.questions_file)
    print(f"loaded {len(docs)} questions", flush=True)

    done = load_done_keys(args.output)
    todo = [d for d in docs if doc_key(d) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"todo: {len(todo)} (done: {len(done)})", flush=True)
    if not todo:
        return

    if args.dry_run:
        for d in todo[:3]:
            msgs = build_request_messages(d, args._tokenizer, args._max_content_tokens)
            n_chars = sum(len(m["content"]) for m in msgs)
            print(f"  [{doc_key(d)}] {len(msgs)} messages, {n_chars} chars, "
                  f"last: {msgs[-1]['content'][:80]!r}", flush=True)
        print("dry-run: no requests sent", flush=True)
        return

    extra_body = json.loads(args.thinking_json) if args.thinking_json else {}
    try:
        from openai import AsyncOpenAI
    except ImportError:
        sys.exit("openai package is required: pip install openai")
    client = AsyncOpenAI(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
        max_retries=0,  # retries handled manually
    )
    sem = asyncio.Semaphore(args.concurrency)
    out = open(args.output, "a", encoding="utf-8")
    t0 = time.time()
    counter = {"n": 0}

    async def worker(doc):
        async with sem:
            row = await generate_one(client, doc, args, extra_body)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        counter["n"] += 1
        n = counter["n"]
        if n % 5 == 0 or n == len(todo):
            rate = n / (time.time() - t0) * 60
            print(f"progress {n}/{len(todo)} ({rate:.1f} q/min)", flush=True)

    await asyncio.gather(*(worker(d) for d in todo))
    out.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model name served by the endpoint")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=os.environ.get("BEAM_API_KEY", ""),
                        help="API key (default: env BEAM_API_KEY)")
    parser.add_argument("--output", default="answers.jsonl", help="output JSONL path")
    parser.add_argument("--chats-file", default=CHATS_FILE)
    parser.add_argument("--questions-file", default=QUESTIONS_FILE)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3600,
                        help="per-request timeout seconds (1M prompts are slow)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--thinking-json", default="",
                        help='extra request body JSON, e.g. '
                             "'{\"thinking\":{\"type\":\"enabled\",\"keep\":\"all\",\"effort\":\"max\"}}'")
    parser.add_argument("--tokenizer", default=BUILTIN_TOKENIZER_DIR,
                        help="HF id or local path of the tokenizer used for "
                             "truncation (default: the bundled Kimi-K2.6 "
                             "tokenizer). Pass an empty string to disable "
                             "truncation.")
    parser.add_argument("--max-context-tokens", type=int, default=1048576,
                        help="model context size; content budget = "
                             "max-context-tokens - max-tokens - 5000")
    parser.add_argument("--limit", type=int, default=0, help="only run first N questions")
    parser.add_argument("--dry-run", action="store_true",
                        help="build messages and print stats without sending requests")
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        sys.exit("missing API key: pass --api-key or export BEAM_API_KEY")

    args._tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            sys.exit("transformers package is required for truncation: "
                     "pip install transformers (or pass --tokenizer '' to disable)")
        try:
            args._tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                            trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"failed to load tokenizer {args.tokenizer!r}: {e!r}")
    args._max_content_tokens = args.max_context_tokens - args.max_tokens - RESERVE_TOKENS

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
