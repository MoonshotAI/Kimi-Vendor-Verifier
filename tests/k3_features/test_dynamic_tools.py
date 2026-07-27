"""Dynamically Loaded Tools (new_model).

Tools may be declared inside a system message, in addition to the request
level (global) tool declare. The tool description format is identical to the
global one and must carry the full tool info; the global declare carries no
`defer` marker.

These tests are written against the online chat-completions endpoint. Because
`tools` inside a message is not part of the standard OpenAI message schema
(the openai SDK would drop it), the requests are sent via the raw httpx
`hclient` fixture and the raw JSON response is inspected.

Tests are grouped as:
  1. Positive: dynamic tools callable from various system positions / forms.
  2. Positive: interaction between global and dynamic tools.
  3. Negative: role constraints (dynamic tools only allowed in system).
  4. Negative: tool definition validation (missing / invalid fields).
  5. Negative: duplicate tool names.
  6. Negative: tool role without tool_call_id.
"""

from typing import Any

import httpx
import pytest



WEATHER_TOOL: dict[str, Any] = {
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


def _fn(name: str, description: str = "A tool.") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _nested_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "nested_tool",
            "description": "A tool with a nested parameter schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "object",
                        "properties": {
                            "lat": {"type": "number"},
                            "lon": {"type": "number"},
                        },
                    },
                },
            },
        },
    }


def _bogus_type_tool() -> dict[str, Any]:
    return {"type": "bogus", "function": {"name": "x"}}


def _post(hclient: httpx.Client, model: str, payload: dict[str, Any]) -> httpx.Response:
    return hclient.post(
        "/chat/completions",
        json={"model": model, **payload},
        timeout=120,
    )


# =============================================================================
# 1. Positive: dynamic tools callable from various system positions / forms
#
# NOTE: new_model only supports tool_choice = "auto" | "none" | "required". The
# named-function form ({"type": "function", ...}) is NOT supported, so these
# tests force tool call(s) with "required" instead (required = must call at
# least one tool, possibly more). When a single tool is available we assert its
# exact name; when several are available we assert the first called tool is one
# of them.
# =============================================================================

@pytest.mark.smoke_test
def test_dynamic_tool_in_system_callable(hclient: httpx.Client, model: str):
    """A tool declared inside a system message can be selected by the model."""
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [WEATHER_TOOL]},
                {"role": "user", "content": "what is the weather in beijing?"},
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    tool_calls = choice["message"].get("tool_calls") or []
    assert tool_calls, f"expected a tool_call, got: {choice['message']}"
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_dynamic_tool_in_subsequent_system_message(hclient: httpx.Client, model: str):
    """A dynamic tool may appear in a system message that is not the first
    message of the conversation."""
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "system", "content": "", "tools": [WEATHER_TOOL]},
                {"role": "user", "content": "what is the weather in beijing?"},
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    tool_calls = choice["message"].get("tool_calls") or []
    assert tool_calls and tool_calls[0]["function"]["name"] == "get_weather"


def test_dynamic_tool_as_last_message(hclient: httpx.Client, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "user", "content": "what is the weather in beijing?"},
                {"role": "system", "content": "", "tools": [WEATHER_TOOL]},
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    tool_calls = choice["message"].get("tool_calls") or []
    assert tool_calls and tool_calls[0]["function"]["name"] == "get_weather"


def test_three_dynamic_tools_in_one_message(hclient: httpx.Client, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {
                    "role": "system",
                    "content": "",
                    "tools": [_fn("get_weather"), _fn("get_time"), _fn("get_news")],
                },
                {"role": "user", "content": "what is the weather in beijing?"},
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    tool_calls = choice["message"].get("tool_calls") or []
    assert tool_calls, f"expected a tool_call, got: {choice['message']}"
    assert tool_calls[0]["function"]["name"] in {"get_weather", "get_time", "get_news"}, (
        f"expected one of the declared tools to be called, got: {tool_calls[0]}"
    )


@pytest.mark.smoke_test
def test_dynamic_tool_with_nested_schema(hclient: httpx.Client, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [_nested_tool()]},
                {"role": "user", "content": "use the tool"},
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    tool_calls = choice["message"].get("tool_calls") or []
    assert tool_calls and tool_calls[0]["function"]["name"] == "nested_tool"


def test_two_dynamic_messages_with_distinct_tools(hclient: httpx.Client, model: str):
    """Two system messages each declaring a different tool are both accepted."""
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [_fn("get_weather")]},
                {"role": "system", "content": "", "tools": [_fn("get_time")]},
                {"role": "user", "content": "what is the weather in beijing?"},
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    tool_calls = choice["message"].get("tool_calls") or []
    assert tool_calls, f"expected a tool_call, got: {choice['message']}"
    assert tool_calls[0]["function"]["name"] in {"get_weather", "get_time"}, (
        f"expected one of the declared tools to be called, got: {tool_calls[0]}"
    )


def test_dynamic_tool_strict_false(hclient: httpx.Client, model: str):
    """A dynamic tool with strict=false is accepted (relaxed parameter
    validation)."""
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather of a city.",
            "strict": False,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [tool]},
                {"role": "user", "content": "what is the weather in beijing?"},
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    tool_calls = choice["message"].get("tool_calls") or []
    assert tool_calls and tool_calls[0]["function"]["name"] == "get_weather"


# =============================================================================
# 2. Positive: interaction between global and dynamic tools
# =============================================================================

@pytest.mark.smoke_test
@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_global_and_dynamic_tools_coexist(hclient: httpx.Client, model: str):
    """A request level (global) tool and a dynamic tool have equal status:
    under tool_choice='required' the model can call the dynamic tool even when
    a global tool is also present. The prompt is weather-related so the dynamic
    get_weather tool is the natural choice."""
    global_tool = _fn("get_stock_price", "Get the stock price of a company.")
    dynamic_tool = WEATHER_TOOL  # get_weather, the only weather-related tool
    response = _post(
        hclient,
        model,
        {
            "tools": [global_tool],
            "messages": [
                {"role": "system", "content": "", "tools": [dynamic_tool]},
                {"role": "user", "content": "what is the weather in beijing?"},
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    tool_calls = choice["message"].get("tool_calls") or []
    assert tool_calls, f"expected a tool_call, got: {choice['message']}"
    called = {tc["function"]["name"] for tc in tool_calls}
    assert "get_weather" in called, (
        f"expected the dynamic get_weather tool to be called under 'required', "
        f"got called tools: {called}"
    )


def test_tool_choice_required_with_only_dynamic_tools(hclient: httpx.Client, model: str):
    """tool_choice='required' forces a call even when tools are only dynamic."""
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [WEATHER_TOOL]},
                {"role": "user", "content": "what is the weather in beijing?"},
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    assert choice["message"].get("tool_calls"), f"expected a tool_call, got: {choice['message']}"


# =============================================================================
# 3. Negative: role constraints (dynamic tools only allowed in system)
# =============================================================================


def test_dynamic_tool_in_user_message_rejected(hclient: httpx.Client, model: str):
    """dynamic tools are only allowed in the system role (not user)."""
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "user", "content": "hi", "tools": [WEATHER_TOOL]},
                {"role": "user", "content": "what is the weather in beijing?"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for a dynamic tool in a user message, "
        f"got {response.status_code}: {response.text}"
    )


def test_dynamic_tool_in_assistant_message_rejected(hclient: httpx.Client, model: str):
    """dynamic tools are only allowed in the system role (not assistant)."""
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello", "tools": [WEATHER_TOOL]},
                {"role": "user", "content": "what is the weather in beijing?"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for a dynamic tool in an assistant message, "
        f"got {response.status_code}: {response.text}"
    )


def test_content_and_dynamic_tools_nonempty_rejected(hclient: httpx.Client, model: str):
    """A system message with both non-empty content and dynamic tools is
    rejected."""
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "not empty", "tools": [WEATHER_TOOL]},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for non-empty content together with dynamic tools, "
        f"got {response.status_code}: {response.text}"
    )


# =============================================================================
# 4. Negative: tool definition validation (missing / invalid fields)
# =============================================================================


def _tool_without(field: str) -> dict[str, Any]:
    """Build a valid tool then remove a single top-level field from it.

    field is one of: "type", "function", "name", "parameters".
    """
    tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "good_tool",
            "description": "A tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    if field in ("name", "parameters"):
        del tool["function"][field]
    else:
        del tool[field]
    return tool


@pytest.mark.parametrize(
    "missing",
    ["type", "function", "name"],
    ids=["missing_type", "missing_function", "missing_name"],
)
def test_dynamic_tool_missing_required_field_rejected(hclient: httpx.Client, missing: str, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [_tool_without(missing)]},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for a dynamic tool missing {missing!r}, "
        f"got {response.status_code}: {response.text}"
    )


@pytest.mark.parametrize(
    "bad_tool",
    [
        _fn("1bad_name"),
        _fn("bad@name"),
        _fn(""),
        _fn("a" * 257),
    ],
    ids=["starts_with_number", "special_char", "empty", "too_long_257"],
)
def test_invalid_dynamic_tool_name_rejected(hclient: httpx.Client, bad_tool: dict[str, Any], model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [bad_tool]},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for invalid dynamic tool {bad_tool['function']['name']!r}, "
        f"got {response.status_code}: {response.text}"
    )


def test_unsupported_tool_type_rejected(hclient: httpx.Client, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [_bogus_type_tool()]},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for an unsupported tool type, got {response.status_code}: {response.text}"
    )


def test_mixed_valid_and_bogus_type_tools_rejected(hclient: httpx.Client, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {
                    "role": "system",
                    "content": "",
                    "tools": [_fn("good_tool"), _bogus_type_tool()],
                },
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for a request mixing valid and bogus-type tools, "
        f"got {response.status_code}: {response.text}"
    )


def test_dynamic_tools_not_array_rejected(hclient: httpx.Client, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": {"type": "function"}},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 when tools is not an array, got {response.status_code}: {response.text}"
    )


def test_dynamic_tools_item_not_object_rejected(hclient: httpx.Client, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [None]},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 when a tools item is not an object, "
        f"got {response.status_code}: {response.text}"
    )


# =============================================================================
# 5. Negative: duplicate tool names
# =============================================================================


def test_duplicate_dynamic_tool_names_rejected(hclient: httpx.Client, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [_fn("dup"), _fn("dup")]},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for duplicate dynamic tool names, got {response.status_code}: {response.text}"
    )


def test_duplicate_tool_names_across_dynamic_messages_rejected(hclient: httpx.Client, model: str):
    """Two distinct dynamic messages declaring tools with the same name."""
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "system", "content": "", "tools": [_fn("dup")]},
                {"role": "system", "content": "", "tools": [_fn("dup")]},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for duplicate tool names across dynamic messages, "
        f"got {response.status_code}: {response.text}"
    )

@pytest.mark.smoke_test
def test_duplicate_tool_name_between_global_and_dynamic_rejected(hclient: httpx.Client, model: str):
    """A request level tool and a dynamic tool sharing the same name."""
    response = _post(
        hclient,
        model,
        {
            "tools": [_fn("dup")],
            "messages": [
                {"role": "system", "content": "", "tools": [_fn("dup")]},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for duplicate tool name between global and dynamic tools, "
        f"got {response.status_code}: {response.text}"
    )


# =============================================================================
# 6. Negative: tool role without tool_call_id
# =============================================================================


def test_tool_role_without_tool_call_id_rejected(hclient: httpx.Client, model: str):
    response = _post(
        hclient,
        model,
        {
            "messages": [
                {"role": "tool", "content": "some tool result"},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for a tool message without tool_call_id, "
        f"got {response.status_code}: {response.text}"
    )
