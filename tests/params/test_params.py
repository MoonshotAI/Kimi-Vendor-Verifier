"""API parameter constraint tests.

Verifies that immutable parameters use their default values and that wrong
values are rejected by the vendor API. Covers both thinking and non-thinking
modes.Only test with think-mode kimi, not opensource
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any

import openai
import pytest

from tests.helper import get_thinking_extra_body

@dataclass
class ParamSpec:
    name: str
    think_default: Any
    non_think_default: Any
    wrong_value: Any


IMMUTABLE_PARAMS: list[ParamSpec] = [
    ParamSpec("temperature", 0.6, 0.6, 1.1),
    ParamSpec("temperature", 0.0, 0.6, 2.0),
    ParamSpec("temperature", 1.0, 0.6, -0.1),
    ParamSpec("top_p", 0.95, 0.95, 0.8),
    ParamSpec("presence_penalty", 0, 0, 0.5),
    ParamSpec("frequency_penalty", 0, 0, 0.5),
    ParamSpec("n", 1, 1, 2),
]


PARAM_IDS = [
    f"{param.name}={param.wrong_value}"
    for param in IMMUTABLE_PARAMS
]


def _make_request(
    client: openai.Client,
    model: str,
    thinking: bool,
    think_mode: str,
    extra_params: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'OK' and nothing else."}],
        "extra_body": get_thinking_extra_body(thinking, think_mode, effort="max"),
    }
    if extra_params:
        kwargs.update(extra_params)

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content[:50] if response.choices else "no content"
        return True, f"OK: {content}"
    except openai.BadRequestError as exc:
        return False, f"Rejected(400): {exc.message}"
    except Exception as exc:
        return False, f"Error: {type(exc).__name__}: {exc}"


@pytest.mark.parametrize("thinking", [False, True], ids=["non-thinking", "thinking"])
def test_no_param_succeeds(
    client: openai.Client,
    model: str,
    thinking: bool,
    think_mode: str,
) -> None:
    success, msg = _make_request(client, model, thinking, think_mode)
    assert success, f"request without optional params failed: {msg}"


@pytest.mark.parametrize("thinking", [False, True], ids=["non-thinking", "thinking"])
def test_default_param_accepted(
    client: openai.Client,
    model: str,
    thinking: bool,
    think_mode: str,
) -> None:
    for param in IMMUTABLE_PARAMS:
        default_value = param.think_default if thinking else param.non_think_default
        success, msg = _make_request(
            client, model, thinking, think_mode, {param.name: default_value}
        )
        assert success, f"{param.name}={default_value} should be accepted: {msg}"


@pytest.mark.parametrize("param", IMMUTABLE_PARAMS, ids=PARAM_IDS)
@pytest.mark.parametrize("thinking", [False, True], ids=["non-thinking", "thinking"])
def test_wrong_param_rejected(
    client: openai.Client,
    model: str,
    thinking: bool,
    think_mode: str,
    param: ParamSpec,
) -> None:
    default_value = param.think_default if thinking else param.non_think_default
    if param.wrong_value == default_value:
        pytest.skip("wrong_value equals default_value")

    success, msg = _make_request(
        client, model, thinking, think_mode, {param.name: param.wrong_value}
    )
    assert not success, (
        f"{param.name}={param.wrong_value} should be rejected, but request succeeded: {msg}"
    )


# ---------------------------------------------------------------------------
# tool name length: names up to 128 chars (the limit enforced by the official
# api.msh.team gateway) must be accepted and callable. This is model-agnostic
# and applies to k2.6/k3 alike.
# ---------------------------------------------------------------------------


def _long_named_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get the weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def test_tool_name_128_chars_accepted(client: openai.Client, model: str) -> None:
    """A tool name at the 128-char maximum is accepted and callable."""
    name = "tool" + "a" * 124  # exactly 128 chars
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "what is the weather in beijing?"}],
        tools=[_long_named_tool(name)],
        tool_choice="required",
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls", choice
    tool_calls = choice.message.tool_calls or []
    assert tool_calls, f"expected a tool_call, got: {choice.message}"
    assert tool_calls[0].function.name == name
