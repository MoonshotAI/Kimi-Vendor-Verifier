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

from tests.helper import token_kwarg, get_thinking_extra_body

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
    think_mode: str,
    effort: str | None = None,
    reasoning_effort: str | None = None,
    thinking: bool = True,
) -> int:
    """Run a completion with the given thinking.effort / reasoning_effort
    (None = omit the field) and return the length of the reasoning_content, return 0
     when thinking false."""

    if think_mode == "kimi":
        extra_body = get_thinking_extra_body(thinking, think_mode, effort)
        # effor for thinking object, reasoning_effort for top level parameter
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort
    else:
        #opensource effort, reasoning_effort map to the same field.
        extra_body = get_thinking_extra_body(
            thinking, think_mode, effort or reasoning_effort
        )

    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        **token_kwarg(32 * 1024),
        extra_body=extra_body,
    )
    reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
    return len(reasoning or "")


def _ask_effort(
    client: openai.Client,
    model: str,
    think_mode: str,
    *,
    effort: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Ask the model to report its effective effort and return the raw content.

    The model reads its effective effort from the system prompt, so this is a
    direct (and more reliable than reasoning-length) way to check which effort
    is in effect. We do NOT parse JSON; the caller checks containment with `in`.
    """

    extra_body = get_thinking_extra_body(True, think_mode, effort)
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Which thinking_effort is the system invoked with? (low/high/max)"
                ),
            },
        ],
        **token_kwarg(256),
        extra_body=extra_body,
    )
    return completion.choices[0].message.content or ""


def test_basic_enabled_returns_reasoning(
    client: openai.Client, model: str, think_mode: str
):
    """Most basic case: thinking enabled with omitted effort returns
    both reasoning_content and content."""
    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        **token_kwarg(32 * 1024),
        extra_body=get_thinking_extra_body(True, think_mode),
    )
    choice = completion.choices[0]
    assert hasattr(choice.message, "reasoning_content"), (
        f"expected reasoning_content when thinking is enabled, got: {choice.message}"
    )
    assert choice.message.content, f"expected content, got: {choice.message}"


def test_default_type_is_enabled(client: openai.Client, model: str, think_mode: str):
    """Omitting thinking.type defaults to 'enabled', so reasoning_content is
    returned. Open-source mode omits the param entirely (server default)."""
    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        **token_kwarg(32 * 1024),
        extra_body=(
            {} if think_mode == "opensource" else {"thinking": {"keep": "all"}}
        ),
    )
    choice = completion.choices[0]
    assert hasattr(choice.message, "reasoning_content"), (
        f"expected reasoning_content when thinking.type is omitted (default "
        f"enabled), got: {choice.message}"
    )
    assert choice.message.content, f"expected content, got: {choice.message}"


def test_effort_omitted_returns_reasoning(
    client: openai.Client, model: str, think_mode: str
):
    """Omitting effort (default = max) with thinking enabled still returns
    reasoning_content."""
    completion = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        **token_kwarg(32 * 1024),
        extra_body=get_thinking_extra_body(True, think_mode),
    )
    choice = completion.choices[0]
    assert hasattr(choice.message, "reasoning_content"), (
        f"expected reasoning_content with default (max) effort, got: {choice.message}"
    )
    assert choice.message.content, f"expected content, got: {choice.message}"

    report = _ask_effort(client, model, think_mode)
    assert "max" in report, f"expected 'max' in effort report, got: {report!r}"


@pytest.mark.skip(reason="temporarily skipped: reasoning length is noisy")
@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_effort_reasoning_length_increases(
    client: openai.Client, model: str, think_mode: str
):
    """Higher effort should produce more reasoning: low < high < max. Verified
    both by reasoning length and by directly asking the model its effort."""
    low, high, max_ = (
        _reasoning_length(client, model, think_mode, effort=e) for e in _EFFORTS
    )
    assert low < high < max_, (
        f"expected reasoning length low < high < max, "
        f"got low={low}, high={high}, max={max_}"
    )
    assert "low" in _ask_effort(client, model, think_mode, effort="low"), "expected 'low' in effort report"
    assert "high" in _ask_effort(client, model, think_mode, effort="high"), "expected 'high' in effort report"
    assert "max" in _ask_effort(client, model, think_mode, effort="max"), "expected 'max' in effort report"


@pytest.mark.skip(reason="temporarily skipped: reasoning length is noisy")
@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_effort_default_closest_to_max(client: openai.Client, model: str, think_mode: str):
    """The default (effort omitted) should behave like effort='max', i.e. its
    reasoning length is closest to that of 'max'."""
    default = _reasoning_length(client, model, think_mode)
    low, high, max_ = (
        _reasoning_length(client, model, think_mode, effort=e) for e in _EFFORTS
    )
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

    report = _ask_effort(client, model, think_mode)
    assert "max" in report, f"expected 'max' in effort report, got: {report!r}"


@pytest.mark.flaky(reruns=5, reruns_delay=1)
def test_reasoning_effort_ignored_when_effort_present(
    client: openai.Client, model: str, think_mode: str, require_platform_server
):
    """When thinking.effort is present, reasoning_effort is ignored:
    thinking.effort=low + reasoning_effort=max should behave like effort=low,
    not effort=max. Verified both by reasoning length and by directly asking the
    model its effective effort. (Kimi-platform precedence; open-source servers
    only have reasoning_effort, so the conflict cannot arise.)"""
    low_only = _reasoning_length(client, model, think_mode, effort="low")
    max_only = _reasoning_length(client, model, think_mode, effort="max")
    with_ignored = _reasoning_length(
        client, model, think_mode, effort="low", reasoning_effort="max"
    )

    assert with_ignored < max_only, (
        f"reasoning_effort should be ignored when thinking.effort present: "
        f"with_ignored={with_ignored}, max_only={max_only}"
    )

    report = _ask_effort(client, model, think_mode, effort="low", reasoning_effort="max")
    assert "low" in report, f"expected 'low' in effort report, got: {report!r}"


@pytest.mark.flaky(reruns=5, reruns_delay=1)
def test_reasoning_effort_effective_when_effort_absent(
    client: openai.Client, model: str, think_mode: str
):
    """When thinking.effort is absent, reasoning_effort takes effect:
    reasoning_effort=max should produce more reasoning than
    reasoning_effort=low. (Verified by reasoning length only, since the model
    does not report reasoning_effort in the system prompt.)"""
    low_reasoning = _reasoning_length(client, model, think_mode, reasoning_effort="low")
    max_reasoning = _reasoning_length(client, model, think_mode, reasoning_effort="max")
    assert max_reasoning > low_reasoning, (
        f"reasoning_effort should take effect when thinking.effort absent: "
        f"max_reasoning={max_reasoning}, low_reasoning={low_reasoning}"
    )

    report = _ask_effort(client, model, think_mode, reasoning_effort="max")
    assert "max" in report, f"expected 'max' in effort report, got: {report!r}"


def test_effort_stream_returns_reasoning_chunks(
    client: openai.Client, model: str, think_mode: str
):
    stream = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        **token_kwarg(32 * 1024),
        stream=True,
        extra_body=get_thinking_extra_body(True, think_mode, "high"),
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
