import os
import uuid
from typing import Any, cast

import httpx
from httpcore import ReadError as HttpcoreReadError
from httpcore import RemoteProtocolError
from openai import APIConnectionError, APIStatusError, RateLimitError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from tenacity import wait_exponential_jitter
from typing_extensions import override

from inspect_ai.model import GenerateConfig, modelapi
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

# Unlimited read timeout for thinking mode (model may think for a long time)
STREAM_TIMEOUT = httpx.Timeout(timeout=None, connect=60.0)

RETRYABLE_READ_ERRORS = (
    HttpcoreReadError,
    RemoteProtocolError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


class KimiAPI(OpenAICompatibleAPI):
    """Kimi API provider with streaming and custom retry."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        stream: bool = False,
        **model_args,
    ) -> None:
        # Default http_client has read=600s, too short for thinking mode
        if "http_client" not in model_args:
            model_args["http_client"] = httpx.AsyncClient(timeout=STREAM_TIMEOUT)

        super().__init__(
            model_name=model_name,
            base_url=base_url
            or os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            api_key=api_key or os.environ.get("KIMI_API_KEY"),
            config=config,
            service="kimi",
            stream=stream,
            **model_args,
        )

    @override
    async def _generate_completion(
        self, request: dict[str, Any], config: GenerateConfig
    ) -> ChatCompletion:
        if self.stream:
            request["stream"] = True
            request["stream_options"] = {"include_usage": True}
            return await self._stream_completion(request)
        return cast(
            ChatCompletion, await self.client.chat.completions.create(**request)
        )

    async def _stream_completion(self, request: dict[str, Any]) -> ChatCompletion:
        """Accumulate stream chunks into ChatCompletion."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, Any]] = {}
        usage = None
        model = ""
        finish_reason = None
        completion_id = ""
        created = 0
        started = False

        response = await self.client.chat.completions.create(**request)
        try:
            async for chunk in response:
                started = True
                if chunk.id:
                    completion_id = chunk.id
                if chunk.created:
                    created = chunk.created
                if chunk.model:
                    model = chunk.model
                if chunk.usage:
                    usage = chunk.usage

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if rc := getattr(delta, "reasoning_content", None):
                    if isinstance(rc, str):
                        reasoning_parts.append(rc)

                if delta.content:
                    content_parts.append(delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index is None:
                            continue
                        idx = tc.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.id or str(uuid.uuid4()),
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_map[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_map[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                if hasattr(choice, "usage") and choice.usage:
                    usage = choice.usage

        except RETRYABLE_READ_ERRORS as e:
            # Once started, don't retry - would cause duplicate requests
            if started:
                print(f"[STREAM_INTERRUPTED] {type(e).__name__}, finish_reason={finish_reason}")
            else:
                print(f"[CONNECTION_ERROR] {type(e).__name__}, retrying")
                raise

        # Build tool_calls
        tool_calls: list[ChatCompletionMessageToolCall] | None = None
        if tool_calls_map:
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type="function",
                    function=Function(name=tc["name"], arguments=tc["arguments"]),
                )
                for tc in sorted(tool_calls_map.values(), key=lambda x: x["id"])
            ]

        # Build message
        message_kwargs: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": tool_calls,
        }
        if reasoning_parts:
            message_kwargs["reasoning_content"] = "".join(reasoning_parts)

        if finish_reason == "length":
            print(f"[LENGTH] id={completion_id}, usage={usage}")

        return ChatCompletion(
            id=completion_id or "stream",
            model=model,
            object="chat.completion",
            created=created,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(**message_kwargs),
                    finish_reason=finish_reason or "stop",
                )
            ],
            usage=usage,
        )

    @override
    def should_retry(self, ex: BaseException) -> bool:
        """Retry on 429 and connection errors."""
        if isinstance(ex, RateLimitError):
            print(f"[RETRY] RateLimitError: {ex}")
            return True
        if isinstance(ex, APIStatusError) and ex.status_code == 429:
            print(f"[RETRY] 429: {ex}")
            return True
        if isinstance(ex, (APIConnectionError, *RETRYABLE_READ_ERRORS)):
            print(f"[RETRY] {type(ex).__name__}: {ex}")
            return True
        print(f"[NO_RETRY] {type(ex).__name__}: {ex}")
        return False

    @override
    def retry_wait(self):
        return wait_exponential_jitter(initial=1, max=60, jitter=2)


@modelapi(name="kimi")
def kimi() -> type[KimiAPI]:
    return KimiAPI
