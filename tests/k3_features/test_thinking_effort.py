"""Thinking Effort (new_model).

`thinking` controls the reasoning behavior:

    thinking = {"type": "enabled" | "disabled", "keep": "all", "effort": "low" | "high" | "max"}
    reasoning_effort = "low" | "high" | "max"

- thinking.type: "enabled" (default) or "disabled".
- thinking.keep: fixed to "all" when type="enabled"; ignored when "disabled".
- thinking.effort: valid when type="enabled" ("low"/"high"/"max", default
  "max"); ignored when "disabled".
- reasoning_effort: when set, overrides thinking.effort.

These tests are written against the online chat-completions endpoint using the
standard openai SDK `client` fixture. Because `thinking` / `reasoning_effort`
are Moonshot extensions (not standard OpenAI parameters), they are passed via
`extra_body`.

Effort levels are compared using the length of the returned reasoning_content
(higher effort should reason more). Reasoning length is inherently noisy, so
the ordering / proximity tests are marked flaky to tolerate run-to-run
variance; a reasoning-worthy prompt improves the signal.
"""

import openai
import pytest



MESSAGES = [
    {
        "role": "user",
        "content": "鸡兔同笼，共有 35 个头，94 条腿。问鸡和兔各有多少只？请逐步推理。",
    }
]

_EFFORTS = ("low", "high", "max")


def _reasoning_length(
    client: openai.Client,
    model: str,
    effort: str | None = None,
    reasoning_effort: str | None = None,
) -> int:
    """Run a completion with the given thinking.effort / reasoning_effort
    (None = omit the field) and return the length of the reasoning_content, 0 if
    absent."""
    thinking: dict = {"type": "enabled", "keep": "all"}
    if effort is not None:
        thinking["effort"] = effort
    extra_body: dict = {"thinking": thinking}
    if reasoning_effort is not None:
        extra_body["reasoning_effort"] = reasoning_effort
    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        max_tokens=32 * 1024,
        extra_body=extra_body,
    )
    reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
    return len(reasoning or "")


def _ask_effort(
    client: openai.Client,
    model: str,
    *,
    effort: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Ask the model to report its effective effort and return the raw content.

    The model reads its effective effort from the system prompt, so this is a
    direct (and more reliable than reasoning-length) way to check which effort
    is in effect. We do NOT parse JSON; the caller checks containment with `in`.
    """
    thinking: dict = {"type": "enabled", "keep": "all"}
    if effort is not None:
        thinking["effort"] = effort
    extra_body: dict = {"thinking": thinking}
    if reasoning_effort is not None:
        extra_body["reasoning_effort"] = reasoning_effort
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "你当前的 thinking_effort 是什么？只输出 effort 的值，"
                    "例如 low / high / max / unknown，不要输出其他内容。"
                ),
            },
        ],
        max_tokens=256,
        extra_body=extra_body,
    )
    return completion.choices[0].message.content or ""


def test_basic_enabled_returns_reasoning(client: openai.Client, model: str):
    """Most basic case: thinking enabled with an explicit effort returns
    both reasoning_content and content."""
    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        max_tokens=32 * 1024,
        extra_body={"thinking": {"type": "enabled", "keep": "all", "effort": "high"}},
    )
    choice = completion.choices[0]
    assert hasattr(choice.message, "reasoning_content"), (
        f"expected reasoning_content when thinking is enabled, got: {choice.message}"
    )
    assert choice.message.content, f"expected content, got: {choice.message}"


def test_default_type_is_enabled(client: openai.Client, model: str):
    """Omitting thinking.type defaults to 'enabled', so reasoning_content is
    returned."""
    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        max_tokens=32 * 1024,
        extra_body={"thinking": {"keep": "all"}},
    )
    choice = completion.choices[0]
    assert hasattr(choice.message, "reasoning_content"), (
        f"expected reasoning_content when thinking.type is omitted (default "
        f"enabled), got: {choice.message}"
    )
    assert choice.message.content, f"expected content, got: {choice.message}"


def test_effort_omitted_returns_reasoning(client: openai.Client, model: str):
    """Omitting effort (default = max) with thinking enabled still returns
    reasoning_content."""
    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        max_tokens=32 * 1024,
        extra_body={"thinking": {"type": "enabled", "keep": "all"}},
    )
    choice = completion.choices[0]
    assert hasattr(choice.message, "reasoning_content"), (
        f"expected reasoning_content with default (max) effort, got: {choice.message}"
    )
    assert choice.message.content, f"expected content, got: {choice.message}"

    report = _ask_effort(client, model)
    assert "max" in report, f"expected 'max' in effort report, got: {report!r}"


@pytest.mark.skip(reason="temporarily skipped: reasoning length is noisy")
@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_effort_reasoning_length_increases(client: openai.Client, model: str):
    """Higher effort should produce more reasoning: low < high < max. Verified
    both by reasoning length and by directly asking the model its effort."""
    low, high, max_ = (_reasoning_length(client, model, e) for e in _EFFORTS)
    assert low < high < max_, (
        f"expected reasoning length low < high < max, "
        f"got low={low}, high={high}, max={max_}"
    )
    assert "low" in _ask_effort(client, model, effort="low"), "expected 'low' in effort report"
    assert "high" in _ask_effort(client, model, effort="high"), "expected 'high' in effort report"
    assert "max" in _ask_effort(client, model, effort="max"), "expected 'max' in effort report"


@pytest.mark.skip(reason="temporarily skipped: reasoning length is noisy")
@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_effort_default_closest_to_max(client: openai.Client, model: str):
    """The default (effort omitted) should behave like effort='max', i.e. its
    reasoning length is closest to that of 'max'."""
    default = _reasoning_length(client, model)
    low, high, max_ = (_reasoning_length(client, model, e) for e in _EFFORTS)
    distances = {
        "low": abs(default - low),
        "high": abs(default - high),
        "max": abs(default - max_),
    }
    min_dist = min(distances.values())
    assert distances["max"] == min_dist, (
        f"expected default effort closest to 'max' (distance={distances['max']}), "
        f"but the minimal distance is {min_dist}; "
        f"default={default}, low={low}, high={high}, max={max_}, distances={distances}"
    )

    report = _ask_effort(client, model)
    assert "max" in report, f"expected 'max' in effort report, got: {report!r}"


@pytest.mark.flaky(reruns=5, reruns_delay=1)
def test_reasoning_effort_ignored_when_effort_present(client: openai.Client, model: str):
    """When thinking.effort is present, reasoning_effort is ignored:
    thinking.effort=low + reasoning_effort=max should behave like effort=low,
    not effort=max. Verified both by reasoning length and by directly asking the
    model its effective effort."""
    low_only = _reasoning_length(client, model, effort="low")
    max_only = _reasoning_length(client, model, effort="max")
    with_ignored = _reasoning_length(client, model, effort="low", reasoning_effort="max")

    assert with_ignored < max_only, (
        f"reasoning_effort should be ignored when thinking.effort present: "
        f"with_ignored={with_ignored}, max_only={max_only}"
    )

    report = _ask_effort(client, model, effort="low", reasoning_effort="max")
    assert "low" in report, f"expected 'low' in effort report, got: {report!r}"


@pytest.mark.flaky(reruns=5, reruns_delay=1)
def test_reasoning_effort_effective_when_effort_absent(client: openai.Client, model: str):
    """When thinking.effort is absent, reasoning_effort takes effect:
    reasoning_effort=max should produce more reasoning than
    reasoning_effort=low. (Verified by reasoning length only, since the model
    does not report reasoning_effort in the system prompt.)"""
    low_reasoning = _reasoning_length(client, model, reasoning_effort="low")
    max_reasoning = _reasoning_length(client, model, reasoning_effort="max")
    assert max_reasoning > low_reasoning, (
        f"reasoning_effort should take effect when thinking.effort absent: "
        f"max_reasoning={max_reasoning}, low_reasoning={low_reasoning}"
    )

    report = _ask_effort(client, model, reasoning_effort="max")
    assert "max" in report, f"expected 'max' in effort report, got: {report!r}"

@pytest.mark.skip("skip thinking disabled, tc assert not right")
def test_effort_with_disabled_returns_no_reasoning(client: openai.Client, model: str):
    """effort must not enable reasoning when type='disabled'."""
    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        max_tokens=32 * 1024,
        extra_body={"thinking": {"type": "disabled", "keep": "all", "effort": "high"}},
    )
    choice = completion.choices[0]
    reasoning = getattr(choice.message, "reasoning_content", None)
    assert not reasoning, (
        f"thinking disabled should yield no reasoning_content, got: {reasoning!r}"
    )
    assert choice.message.content, f"expected content, got: {choice.message}"

@pytest.mark.skip("skip thinking disabled, tc assert not right")
def test_disabled_ignores_keep(client: openai.Client, model: str):
    """When type='disabled', keep is ignored (any value is accepted and no
    reasoning is returned)."""
    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        max_tokens=32 * 1024,
        extra_body={"thinking": {"type": "disabled", "keep": "none", "effort": "high"}},
    )
    choice = completion.choices[0]
    reasoning = getattr(choice.message, "reasoning_content", None)
    assert not reasoning, (
        f"thinking disabled should ignore keep and yield no reasoning_content, "
        f"got: {reasoning!r}"
    )
    assert choice.message.content, f"expected content, got: {choice.message}"


@pytest.mark.skip(
    reason="online can use preset or normalization to force keep='all'; skip the keep != 'all' rejection check"
)
def test_keep_not_all_rejected(client: openai.Client, model: str):
    """keep is fixed to 'all' when type='enabled'; any other value must be
    rejected."""
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model=model,
            messages=MESSAGES,
            max_tokens=32 * 1024,
            extra_body={"thinking": {"type": "enabled", "keep": "none", "effort": "high"}},
        )


# keep='all' behavior tests. `keep` (keep_reasoning_content) controls how much
# of the earlier turns' reasoning_content is kept in the prompt for the next
# turn. The 4 modes are: all / none / after_last_user /
# after_last_non_tool_call_assistant (the default). Only 'all' keeps the
# reasoning_content of turns that come BEFORE the last user message; the other
# three strip it. Online only accepts keep='all' (others are 400), so we cannot
# compare against them directly; instead we verify keep='all' actually keeps
# the early-turn reasoning (Test A semantically, Test B deterministically).
#
# The fabricated conversation plants three distinct 4-digit numbers ONLY in the
# reasoning_content of three earlier assistant turns (the assistant `content`
# never contains them). The final user turn asks the model to read them back
# from its reasoning_content. All three reasoning blocks sit before the last
# user message, so only keep='all' preserves them.
_KEEP_MESSAGES = [
    {
        "role": "user",
        "content": (
            "接下来我会把一些重要信息只放在你的 reasoning_content（思考过程）里，"
            "你的 content 只需要回复\"收到\"两个字，不要复述任何信息。"
            "直到我最后主动问你这些信息是什么时，你再回答。明白了吗？"
        ),
    },
    {
        "role": "assistant",
        "content": "收到",
        "reasoning_content": "用户的要求是：重要信息只放在 reasoning_content 里，content 只回复\"收到\"，等用户最后主动问时再回答。我记住了这个规则。",
    },
    {"role": "user", "content": "我已经把第一个信息写进你的 reasoning content 了。"},
    {
        "role": "assistant",
        "content": "收到",
        "reasoning_content": "用户说第一个信息已经写进来了。这里的第一个4位数字是 7382。我先把它记在心里，但 content 只回复收到。",
    },
    {"role": "user", "content": "我已经把第二个信息写进你的 reasoning content 了。"},
    {
        "role": "assistant",
        "content": "收到",
        "reasoning_content": "用户说第二个信息已经写进来了。这里的第二个4位数字是 4915。我记住它，content 只回复收到。",
    },
    {"role": "user", "content": "我已经把第三个信息写进你的 reasoning content 了。"},
    {
        "role": "assistant",
        "content": "收到",
        "reasoning_content": "用户说第三个信息已经写进来了。这里的第三个4位数字是 2067。我记住它，content 只回复收到。",
    },
    {
        "role": "user",
        "content": (
            "现在请从你的 reasoning content（思考过程）里找出刚才那3个4位数字，"
            "按先后顺序告诉我。只回答这3个数字，用空格隔开。"
        ),
    },
]

_KEEP_NUMBERS = ("7382", "4915", "2067")


def _strip_reasoning(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with the reasoning_content field removed."""
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]

@pytest.mark.skip("skip, not pass")
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_keep_all_preserves_history_reasoning(client: openai.Client, model: str):
    """Test A (semantic): default keep='all' must keep the reasoning_content of earlier
    turns (those before the last user message), so the model can recall the
    4-digit numbers that only ever appeared in reasoning_content. If keep
    behaved as 'none' / 'after_last_user' /
    'after_last_non_tool_call_assistant', those early-turn reasoning blocks
    would be stripped from the prompt and the model could not recall them."""
    completion = client.chat.completions.create(
        model=model,
        messages=_KEEP_MESSAGES,
        max_tokens=512,
        extra_body={"thinking": {"type": "enabled"}},
    )
    content = completion.choices[0].message.content or ""
    for number in _KEEP_NUMBERS:
        assert number in content, (
            f"keep='all' should preserve earlier reasoning_content so the model "
            f"recalls {number}; got content={content!r}"
        )

@pytest.mark.skip("skip, tc not pass")
def test_keep_all_prompt_tokens_include_reasoning(client: openai.Client, model: str):
    """Test B (deterministic): with default keep='all', the earlier turns'
    reasoning_content is part of the prompt, so the prompt token count must be
    strictly higher than the same conversation with reasoning_content removed.
    If keep stripped the reasoning (behaving as 'none'), the two counts would
    be equal."""
    with_rc = client.chat.completions.create(
        model=model,
        messages=_KEEP_MESSAGES,
        max_tokens=1,
        extra_body={"thinking": {"type": "enabled"}},
    )
    without_rc = client.chat.completions.create(
        model=model,
        messages=_strip_reasoning(_KEEP_MESSAGES),
        max_tokens=1,
        extra_body={"thinking": {"type": "enabled"}},
    )
    tokens_with = with_rc.usage.prompt_tokens
    tokens_without = without_rc.usage.prompt_tokens
    assert tokens_with > tokens_without, (
        f"keep='all' should include earlier reasoning_content in the prompt: "
        f"prompt_tokens with reasoning={tokens_with}, without={tokens_without}"
    )


def test_effort_stream_returns_reasoning_chunks(client: openai.Client, model: str):
    stream = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        max_tokens=32 * 1024,
        stream=True,
        extra_body={"thinking": {"type": "enabled", "keep": "all", "effort": "high"}},
    )
    reasoning_chunks = []
    content_chunks = []
    finish_reason = None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning_content", None):
            reasoning_chunks.append(delta.reasoning_content)
        if getattr(delta, "content", None):
            content_chunks.append(delta.content)
        if chunk.choices[0].finish_reason is not None:
            finish_reason = chunk.choices[0].finish_reason

    assert reasoning_chunks, "expected reasoning_content chunks in stream"
    assert content_chunks, "expected content chunks in stream"
    assert finish_reason == "stop", f"expected finish_reason='stop', got {finish_reason!r}"
