"""Stream-mode parametrization for the k3_features API contract tests.

Every test in this directory that uses the `client` (openai SDK) or `hclient`
(raw httpx) fixture is run twice:

  [nostream] the fixtures behave exactly as the ones in tests/conftest.py
             (plain non-streaming requests);
  [stream]   the fixtures return wrappers that issue the same request with
             stream=true, consume the whole SSE chunk stream and accumulate it
             back into a response equivalent to the non-streaming one, so the
             existing assertions run unchanged against streaming behavior.

The wrapping is done here (not in the test files) so the test logic stays
identical between the two modes.
"""

from __future__ import annotations

import json as jsonlib
from typing import Any, Iterator

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run every test that uses `client` or `hclient` in both modes.

    `_stream` is parametrized indirectly: it is not part of any test
    signature, but the `client` / `hclient` fixtures below request it, so the
    parameter reaches them through the fixture closure.
    """
    if {"client", "hclient"} & set(metafunc.fixturenames):
        metafunc.parametrize(
            "_stream",
            [False, True],
            ids=["nostream", "stream"],
            indirect=True,
        )


@pytest.fixture(scope="function")
def _stream(request: pytest.FixtureRequest) -> bool:
    return request.param


# =============================================================================
# SSE chunk accumulation
#
# Both wrappers feed chunk dicts (openai SDK chunks via model_dump(), raw SSE
# `data:` payloads via json.loads) into this accumulator, which rebuilds a
# dict shaped like a non-streaming chat.completion response:
#
#   {"id", "object": "chat.completion", "created", "model",
#    "choices": [{"index": 0,
#                 "message": {"role", "content", "reasoning_content"?,
#                             "tool_calls"?},
#                 "finish_reason"}],
#    "usage"?}
# =============================================================================


class _StreamAccumulator:
    def __init__(self) -> None:
        self.id: str | None = None
        self.created: int | None = None
        self.model: str | None = None
        self.role: str | None = None
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None

    def add(self, chunk: dict[str, Any]) -> None:
        if chunk.get("id"):
            self.id = chunk["id"]
        if chunk.get("created") is not None:
            self.created = chunk["created"]
        if chunk.get("model"):
            self.model = chunk["model"]
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("role"):
                self.role = delta["role"]
            if delta.get("content"):
                self.content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                self.reasoning_parts.append(delta["reasoning_content"])
            for tc in delta.get("tool_calls") or []:
                index = tc.get("index", 0)
                slot = self.tool_calls.setdefault(
                    index,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": None, "arguments": ""},
                    },
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                function = tc.get("function") or {}
                if function.get("name") and not slot["function"]["name"]:
                    slot["function"]["name"] = function["name"]
                if function.get("arguments"):
                    slot["function"]["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def result(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": self.role or "assistant",
            "content": "".join(self.content_parts) or None,
        }
        if self.reasoning_parts:
            message["reasoning_content"] = "".join(self.reasoning_parts)
        if self.tool_calls:
            message["tool_calls"] = [
                self.tool_calls[i] for i in sorted(self.tool_calls)
            ]
        result: dict[str, Any] = {
            "id": self.id or "",
            "object": "chat.completion",
            "created": self.created or 0,
            "model": self.model or "",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self.finish_reason or "stop",
                }
            ],
        }
        if self.usage is not None:
            result["usage"] = self.usage
        return result


# =============================================================================
# openai SDK client wrapper
# =============================================================================


class _StreamingCompletions:
    """chat.completions.create() that streams under the hood.

    Issues the request with stream=True (adding
    stream_options={"include_usage": True} unless the caller set it), consumes
    the whole chunk stream inside create() -- so HTTP errors, which the openai
    SDK raises while iterating the stream, still surface from create() as the
    original exception types (e.g. openai.BadRequestError) -- and returns a
    regular ChatCompletion accumulated from the chunks.

    Calls that already pass stream=True are forwarded untouched, so tests that
    exercise raw streaming keep their original semantics.
    """

    def __init__(self, completions: Any) -> None:
        self._completions = completions

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        if kwargs.get("stream"):
            return self._completions.create(*args, **kwargs)
        kwargs["stream"] = True
        kwargs.setdefault("stream_options", {"include_usage": True})
        stream = self._completions.create(*args, **kwargs)
        accumulator = _StreamAccumulator()
        for chunk in stream:
            accumulator.add(chunk.model_dump())
        return ChatCompletion.model_validate(accumulator.result())


class _StreamingChat:
    def __init__(self, chat: Any) -> None:
        self._chat = chat
        self.completions = _StreamingCompletions(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _StreamingClient:
    """Drop-in wrapper around openai.Client used in stream mode."""

    def __init__(self, client: openai.Client) -> None:
        self._client = client
        self.chat = _StreamingChat(client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# =============================================================================
# httpx client wrapper
# =============================================================================


class _AccumulatedResponse:
    """Minimal response object mirroring a non-streaming JSON response.

    Provides the .status_code / .json() / .text surface the tests use.
    """

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    @property
    def text(self) -> str:
        return jsonlib.dumps(self._payload, ensure_ascii=False)


def _inject_stream(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("stream", True)
    payload.setdefault("stream_options", {"include_usage": True})
    return payload


def _parse_sse(body: str) -> Iterator[dict[str, Any]]:
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        yield jsonlib.loads(data)


class _StreamingHttpClient:
    """Drop-in wrapper around httpx.Client used in stream mode.

    post() injects "stream": true (and stream_options.include_usage) into the
    request payload -- both the json=dict and content=<raw JSON string> forms
    -- then accumulates the SSE response into a dict shaped like the
    non-streaming response JSON. Non-2xx responses are not SSE; they are
    returned untouched.
    """

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def post(self, url: str, **kwargs: Any) -> Any:
        if kwargs.get("json") is not None:
            kwargs["json"] = _inject_stream(kwargs["json"])
        elif kwargs.get("content") is not None:
            payload = _inject_stream(jsonlib.loads(kwargs["content"]))
            kwargs["content"] = jsonlib.dumps(payload, ensure_ascii=False)
        else:
            return self._client.post(url, **kwargs)

        response = self._client.post(url, **kwargs)
        if not (200 <= response.status_code < 300):
            return response

        accumulator = _StreamAccumulator()
        seen_chunks = False
        for chunk in _parse_sse(response.text):
            seen_chunks = True
            accumulator.add(chunk)
        if not seen_chunks:
            # The server ignored stream=true and answered with plain JSON.
            return _AccumulatedResponse(response.status_code, response.json())
        return _AccumulatedResponse(response.status_code, accumulator.result())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# =============================================================================
# Fixture overrides (shadow the ones in tests/conftest.py for this directory)
# =============================================================================


@pytest.fixture(scope="function")
def client(pytestconfig: pytest.Config, key: str, _stream: bool) -> Any:
    sdk_client = openai.Client(
        api_key=key,
        base_url=str(pytestconfig.getoption("base_url")).rstrip("/"),
        timeout=120,
    )
    if _stream:
        return _StreamingClient(sdk_client)
    return sdk_client


@pytest.fixture(scope="function")
def hclient(pytestconfig: pytest.Config, key: str, _stream: bool) -> Any:
    http_client = httpx.Client(
        base_url=str(pytestconfig.getoption("base_url")).rstrip("/"),
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
    )
    if _stream:
        yield _StreamingHttpClient(http_client)
    else:
        yield http_client
    http_client.close()
