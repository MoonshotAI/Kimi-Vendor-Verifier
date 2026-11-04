# Prompt Token Cases

Cases for `tests/prompt_tokens/test_prompt_tokens.py`, which checks that a
vendor's `usage.prompt_tokens` matches the expected constant for each request.

## Case format

One JSON object per line in `cases.jsonl` or `vision_cases.jsonl`:

```json
{
  "id": "assistant_hello",
  "description": "single assistant message, reasoning_effort high, streamed",
  "request": {
    "messages": [{"role": "assistant", "content": "你好"}],
    "max_tokens": 1024,
    "reasoning_effort": "high",
    "temperature": 1,
    "top_p": 0.95,
    "n": 1
  },
  "expected_prompt_tokens": 99
}
```

- `request` is sent to `chat/completions` almost verbatim: the script only
  fills in `model` (from `--model`) and forces `stream=true`. Parameters the
  OpenAI client does not know are forwarded via `extra_body`.
- `expected_prompt_tokens` is the constant the vendor's reported
  `usage.prompt_tokens` must equal exactly. The usage is read from the
  stream's `choices[].usage` (Kimi-style; a top-level chunk `usage` is also
  accepted). Constants are maintained by the case authors (confirmed against
  the reference tokenizer); do not guess them.

## Vision cases

`vision_cases.jsonl` covers requests containing one through ten images at
different landscape, portrait, and square dimensions, including text both
before and after an image content part. It also covers a 3584x3584 image at
the 16384-media-token boundary (16421 total prompt tokens), calculated with
the tokenism MoonViT algorithm using a 65536 input-patch limit. The repository
keeps a single source image at `images/vision_fixture.png`. Case requests refer
to it with a URL such as:

```text
fixture://vision_fixture.png?width=1536&height=1024
```

Immediately before sending a request, the test resizes the source image with
nearest-neighbor sampling, encodes it as PNG, and replaces the fixture URL
with a `data:image/png;base64,...` URL. Consequently, groundtruth must be
collected from the exact materialized request, including image dimensions,
count, content-part order, and `detail` value.

## Partial cases

`cases.partial.jsonl` holds the partial-prefill cases (assistant messages
with `"partial": true`). They are excluded from the default run; run them
explicitly with:

```bash
uv run python verify_prompt_tokens.py --model your-model-id \
    --cases testdata/prompt_token_cases/cases.partial.jsonl
```

## k3 cases

Cases with a `k3_` id prefix are migrated from the `serve/tokenism`
repository (`integrationtest/contract/testdata`, master branch) and share
these rules:

- The request comes from the chat-completions input in
  `cases/<category>/<name>.json`; multimodal, `raw_part`, and interleaved /
  preserved-thinking cases are not migrated.
- `expected_prompt_tokens` is derived from the x4 golden response as
  `totalTokens - len(pendingTokenIds)`.
- The `thinking` request parameter is removed and replaced with
  `reasoning_effort`, taken from `thinking.effort` (`"max"` when unset).
  `reasoning_content` inside messages is kept as-is.
- Cases whose transformed request is identical to another migrated case are
  deduplicated.

## Encoder parity-gap cases

`k3_tool_choice_none_no_tools`, `k3_vision_tool_message_image_content`,
`k3_tool_result_stale_name_rewritten`, and
`k3_tool_result_matches_recent_turn` were authored against the K3 reference
encoder (`encoding_k3.py` on the Kimi-K3 Hugging Face repo) to pin
byte-level behaviours a re-implementation may diverge on: the unconditional
`tool_choice=none` control message, image content parts in tool messages, and
the observable branches of `normalize_xtml_tool_result_messages` (the
tool-`name` rewrite after an id match, and matching against the most recent
assistant turn). Note that the re-sort itself is invisible to token counts
(segments are encoded independently, so swapping two tool messages never
changes the total — `k3_tool_message_reordered` and
`k3_tool_parallel_reordered` therefore cannot detect a missing re-sort).

Their `expected_prompt_tokens` are computed locally: requests are rendered
with `build_chat_segments` and encoded with the K3 tokenizer (`total - 3`
pending channel-seed tokens, same convention as the playground cases); the
pipeline reproduces every pre-existing case's constant exactly, and these
cases pass against the 1P `kimi-k3` deployment. The vision case reuses the
1536x1024 image, whose per-image media expansion (2045 tokens) is derived
from the existing vision cases.

Several sibling probes were dropped because the 1P API does not admit them:
it rejects empty system messages, tool messages without a known
`tool_call_id` (so the unmatched-run and `id`-field-fallback branches are
unreachable over the wire), and its own rendering diverges from the HF
reference on system-message `name` attributes and on `strict` / caller-defined
tool-declaration fields (the 1P drops both).

## Playground cases

Cases migrated from the `xiongziwen/playground` repository
(`tests/cases/chat_template_cases.py`, branch `debug/chattv4`; ids are
renamed from the source `CT-xxx` to descriptive `k3_*` names):

- The request is built from the case's `chat_api_request_messages` (falling
  back to `tokenism_request_messages`) plus its `tools` / `response_format` /
  `tool_choice`; the thinking API field is replaced with `reasoning_effort`,
  taken from `thinking_effort` (`"max"` when unset).
- `expected_prompt_tokens` is computed with the case's chatt golden:
  `len(golden prompt ids) - 3`, where the ids are produced by the chatt
  `MessageTemplateEncoderV4` with the lai_v9 tokenizer, including the
  thinking-effort system message (`"max"` is prepended when the case sets no
  explicit effort) and the assistant generation stub
  (`[open]think[sep]`, 3 tokens). Assistant history messages are counted
  the way the chat-completions path renders them — with an extra empty
  think block (6 tokens) per assistant message (e.g. k3_multi_turn).
- Interleaved / preserved-thinking cases (CT-004, CT-013, CT-014, CT-015,
  CT-017, CT-018) are not migrated.
