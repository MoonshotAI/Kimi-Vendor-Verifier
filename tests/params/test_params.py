"""API parameter constraint tests.

Verifies that immutable parameters use their default values and that wrong
values are rejected by the vendor API. Covers both thinking and non-thinking
modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import openai
import pytest


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


def _get_thinking_extra_body(thinking: bool, think_mode: str) -> dict[str, Any]:
    if think_mode == "none":
        return {}
    if think_mode == "opensource":
        return {"chat_template_kwargs": {"thinking": thinking}}
    return {"thinking": {"type": "enabled" if thinking else "disabled"}}


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
        "extra_body": _get_thinking_extra_body(thinking, think_mode),
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
