"""Response Format (new_model).

The request interface matches the existing models. Per the official Kimi docs
(https://platform.kimi.com/docs/api/chat#body-one-of-0-response-format),
`response_format.type` is one of:

    "text"        default, plain text output.
    "json_object" guarantees a valid JSON object (the prompt should ask for JSON).
    "json_schema" Structured Output constrained by a JSON Schema (recommended).

For json_schema, `json_schema.name` and `json_schema.schema` are required;
`json_schema.strict` defaults to true, in which case the schema must conform to
MFJS (Moonshot Flavored JSON Schema) or the server returns an error / warning;
when false, only valid JSON is guaranteed and the internal structure is not
enforced.

These tests are written against the online chat-completions endpoint. Positive
cases use the standard openai SDK `client` fixture; validation / error cases use
the raw httpx `hclient` fixture so that malformed requests reach the server
unchanged (the SDK could otherwise reject or coerce them client-side).
"""

import json

import httpx
import openai
import pytest



WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temperature": {"type": "number"},
    },
    "required": ["city", "temperature"],
    "additionalProperties": False,
}


def test_text_default(client: openai.Client, model: str):
    """type='text' (the default) returns plain text, not constrained to JSON."""
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "用一句话介绍北京。"}],
        response_format={"type": "text"},
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "stop", choice
    assert isinstance(choice.message.content, str) and choice.message.content, (
        f"expected non-empty text content, got: {choice.message.content!r}"
    )

@pytest.mark.smoke_test
def test_json_object(client: openai.Client, model: str):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Return the weather of Beijing as JSON. "
                    "The response must contain a key named 'city'."
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "stop", choice
    content = choice.message.content
    assert isinstance(content, str) and content, f"expected JSON text, got: {content!r}"

    parsed = json.loads(content)  # raises if not valid JSON
    assert isinstance(parsed, dict), f"expected a JSON object, got: {parsed!r}"
    assert "city" in parsed, f"expected key 'city' in {parsed!r}"

@pytest.mark.smoke_test
def test_json_schema_strict(client: openai.Client, model: str):
    """json_schema with strict=true (the default): output must conform to the
    schema (valid JSON object with the required keys and value types)."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Return the weather of Beijing as JSON following the schema.",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "strict": True,
                "schema": WEATHER_SCHEMA,
            },
        },
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "stop", choice
    content = choice.message.content
    assert isinstance(content, str) and content, f"expected JSON text, got: {content!r}"

    parsed = json.loads(content)
    assert isinstance(parsed, dict), f"expected a JSON object, got: {parsed!r}"
    for key in ("city", "temperature"):
        assert key in parsed, f"expected key {key!r} in {parsed!r}"
    assert isinstance(parsed["city"], str), f"city should be a string: {parsed!r}"
    assert isinstance(parsed["temperature"], (int, float)), (
        f"temperature should be a number: {parsed!r}"
    )

@pytest.mark.smoke_test
def test_json_schema_non_strict(client: openai.Client, model: str):
    """json_schema with strict=false: only a valid JSON object is guaranteed;
    the internal structure is not enforced, so only assert valid JSON."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Return the weather of Beijing as JSON.",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        },
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "stop", choice
    content = choice.message.content
    assert isinstance(content, str) and content, f"expected JSON text, got: {content!r}"

    parsed = json.loads(content)  # only guarantees valid JSON
    assert isinstance(parsed, dict), f"expected a JSON object, got: {parsed!r}"


# ---------------------------------------------------------------------------
# validation: missing / invalid json_schema fields (raw httpx so the server
# validates, not the SDK).
# ---------------------------------------------------------------------------


def test_invalid_response_format_rejected(hclient: httpx.Client, model: str):
    response = hclient.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "bogus"},
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for invalid response_format type, "
        f"got {response.status_code}: {response.text}"
    )


def test_json_schema_missing_name_rejected(hclient: httpx.Client, model: str):
    response = hclient.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            },
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for json_schema missing 'name', "
        f"got {response.status_code}: {response.text}"
    )


def test_json_schema_missing_schema_rejected(hclient: httpx.Client, model: str):
    response = hclient.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "weather"},
            },
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for json_schema missing 'schema', "
        f"got {response.status_code}: {response.text}"
    )


def test_json_schema_missing_json_schema_rejected(hclient: httpx.Client, model: str):
    response = hclient.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_schema"},
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for type=json_schema without a json_schema field, "
        f"got {response.status_code}: {response.text}"
    )


def test_json_schema_strict_non_boolean_rejected(hclient: httpx.Client, model: str):
    response = hclient.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "strict": "yes",
                    "schema": {"type": "object"},
                },
            },
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for non-boolean json_schema.strict, "
        f"got {response.status_code}: {response.text}"
    )


def test_json_schema_schema_not_object_rejected(hclient: httpx.Client, model: str):
    response = hclient.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "weather", "schema": "x"},
            },
        },
    )
    assert response.status_code == 400, (
        f"expected 400 for non-object json_schema.schema, "
        f"got {response.status_code}: {response.text}"
    )


# ---------------------------------------------------------------------------
# prompt token groundtruth (response_format only, no other new_model features).
#
# tokenism returns `total_tokens` (and `pending_tokens`), not a field named
# `prompt_tokens`; the API's `usage.prompt_tokens` corresponds to tokenism's
# `total_tokens - pending_tokens`. The expected prompt token count is computed
# inline in each test as (total - pending), using values measured from a local
# tokenism.
#
# The payloads are sent as the raw JSON strings below (NOT re-serialized from
# Python dicts) so the bytes are byte-for-byte identical to what the local
# tokenism validates -- this is what makes the groundtruth reliable.
# ---------------------------------------------------------------------------

RF_WITHOUT_JSON = r"""{
  "model": "__MODEL__",
  "messages": [
    {"role": "user", "content": "请用 JSON 格式回答：北京是中国的首都吗？答案需要包含键 capital。"}
  ],
  "thinking": {"type": "disabled"}
}"""

RF_WITH_JSON = r"""{
  "model": "__MODEL__",
  "messages": [
    {"role": "user", "content": "请用 JSON 格式回答：北京是中国的首都吗？答案需要包含键 capital。"}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "capital_answer",
      "schema": {
        "type": "object",
        "properties": {
          "capital": {"type": "boolean"}
        },
        "required": ["capital"],
        "additionalProperties": false
      }
    }
  },
  "thinking": {"type": "disabled"}
}"""


def test_response_format_prompt_tokens_without(hclient: httpx.Client, model: str):
    """Prompt token count without response_format matches the tokenism
    groundtruth."""
    expected_prompt_tokens = 36

    response = hclient.post(
        "/chat/completions",
        content=RF_WITHOUT_JSON.replace("__MODEL__", model),
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    assert response.status_code == 200, response.text
    prompt_tokens = response.json()["usage"]["prompt_tokens"]
    assert prompt_tokens == expected_prompt_tokens, (
        f"prompt_tokens={prompt_tokens}, expected={expected_prompt_tokens}"
    )


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_response_format_with(hclient: httpx.Client, model: str):
    """With response_format: prompt token count matches the tokenism groundtruth,
    and the json_schema(capital: bool) output is compact JSON with no literal
    newline (\\n) or tab (\\t) characters. Retried a few times."""
    expected_prompt_tokens = 119

    response = hclient.post(
        "/chat/completions",
        content=RF_WITH_JSON.replace("__MODEL__", model),
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # prompt token count groundtruth
    prompt_tokens = body["usage"]["prompt_tokens"]
    assert prompt_tokens == expected_prompt_tokens, (
        f"prompt_tokens={prompt_tokens}, expected={expected_prompt_tokens}"
    )

    # output is compact JSON with the required key and no whitespace
    content = body["choices"][0]["message"].get("content") or ""
    assert content, f"expected JSON text, got: {content!r}"
    parsed = json.loads(content)
    assert isinstance(parsed, dict) and "capital" in parsed, (
        f"expected key 'capital' in {parsed!r}"
    )
    newline_count = content.count("\n")
    tab_count = content.count("\t")
    assert newline_count == 0, (
        f"expected 0 newlines, got {newline_count}; content={content!r}"
    )
    assert tab_count == 0, (
        f"expected 0 tabs, got {tab_count}; content={content!r}"
    )