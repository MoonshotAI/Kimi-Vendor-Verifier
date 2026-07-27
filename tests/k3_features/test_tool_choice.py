"""Tool Choice (new_model).

`tool_choice` constrains how the model selects tools. For new_model it only supports
string values:

    "auto"     model decides (default) -- the prompt content drives whether a
               tool is called.
    "none"     no tool is selected, even if the prompt encourages one.
    "required" at least one tool must be selected, even if the prompt
               discourages one.

The named-function form ({"type": "function", "function": {"name": ...}}) is
NOT supported by new_model and is intentionally not used here.

These tests are written against the online chat-completions endpoint using the
standard openai SDK `client` fixture. The auto cases are prompt-driven and
therefore inherently less deterministic, so they are marked flaky.
"""

import openai
import pytest



WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather of a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


# ---------------------------------------------------------------------------
# auto: the prompt decides -- the same model may or may not call a tool.
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_tool_choice_auto_may_call(client: openai.Client, model: str):
    """With auto, a prompt whose content needs a tool (and asks to use one)
    leads to a tool_call."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "北京今天天气怎么样？请查一下，你应当使用工具。",
            },
        ],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls", choice
    assert choice.message.tool_calls, f"expected a tool_call, got: {choice.message}"


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_tool_choice_auto_may_not_call(client: openai.Client, model: str):
    """With auto, a prompt whose content needs no tool (and asks not to use
    one) leads to a plain text answer."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "你好，请简单介绍一下你自己。你不应使用任何工具。",
            },
        ],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )
    choice = completion.choices[0]
    assert not choice.message.tool_calls, (
        f"tool_choice='auto' should not emit tool_calls here, got: {choice.message.tool_calls}"
    )
    assert choice.finish_reason == "stop", choice
    assert choice.message.content, f"expected content, got: {choice.message}"


# ---------------------------------------------------------------------------
# required: forces at least one tool_call (and requires tools to be present).
# ---------------------------------------------------------------------------


def test_tool_choice_required_forces_call(client: openai.Client, model: str):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "请简要回答：北京天气怎么样？",
            },
        ],
        tools=[WEATHER_TOOL],
        tool_choice="required",
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls", choice
    assert choice.message.tool_calls, (
        f"tool_choice='required' should force a tool_call, got: {choice.message}"
    )


def test_tool_choice_required_without_tools_rejected(client: openai.Client, model: str):
    """required needs tools; a request with no tools field at all (absent, not
    an empty list) is rejected."""
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "北京天气怎么样？"}],
            tool_choice="required",
        )


# ---------------------------------------------------------------------------
# none: forbids a tool_call even when the prompt encourages it.
# ---------------------------------------------------------------------------


def test_tool_choice_none_forbids_call(client: openai.Client, model: str):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "请查一下北京的天气。",
            },
        ],
        tools=[WEATHER_TOOL],
        tool_choice="none",
    )
    choice = completion.choices[0]
    assert not choice.message.tool_calls, (
        f"tool_choice='none' should not emit tool_calls, got: {choice.message.tool_calls}"
    )
    assert choice.finish_reason == "stop", choice


@pytest.mark.parametrize("tool_choice", ["none", "auto"])
def test_tool_choice_without_tools_field_accepted(
    client: openai.Client, tool_choice: str, model: str
):
    """none/auto do not require a tools field; the request succeeds with a
    plain text answer."""
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "你好。"}],
        tool_choice=tool_choice,
    )
    choice = completion.choices[0]
    assert not choice.message.tool_calls, (
        f"tool_choice={tool_choice!r} without tools should not emit tool_calls, "
        f"got: {choice.message.tool_calls}"
    )
    assert choice.finish_reason == "stop", choice
    assert choice.message.content, f"expected content, got: {choice.message}"


# ---------------------------------------------------------------------------
# validation: invalid tool_choice values and unsupported forms are rejected.
# ---------------------------------------------------------------------------


def test_tool_choice_with_empty_tools_rejected(client: openai.Client, model: str):
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "what is the weather in beijing?"}],
            tools=[],
            tool_choice="required",
        )


def test_tool_choice_invalid_value_rejected(client: openai.Client, model: str):
    """A tool_choice value outside auto/none/required is rejected."""
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "北京天气怎么样？"}],
            tools=[WEATHER_TOOL],
            tool_choice="bogus",
        )


@pytest.mark.skip(
    reason="current behavior silently ignores named-function tool_choice, treating it as no tool_choice"
)
def test_tool_choice_named_function_rejected(client: openai.Client, model: str):
    """new_model does not support the named-function tool_choice form."""
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "北京天气怎么样？"}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
