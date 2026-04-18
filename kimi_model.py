import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
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

from inspect_ai.log import transcript
from inspect_ai.model import GenerateConfig, modelapi
from inspect_ai.model._openai import OpenAIAsyncHttpxClient
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

# Unlimited read timeout for thinking mode (model may think for a long time)
STREAM_TIMEOUT = httpx.Timeout(timeout=None, connect=60.0)
STREAM_LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=100)

RETRYABLE_READ_ERRORS = (
    HttpcoreReadError,
    RemoteProtocolError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


def _default_http_client() -> httpx.AsyncClient:
    """Use inspect_ai/OpenAI client defaults without a hidden pool cap.

    inspect_ai already enforces `max_connections` with its own semaphore, so the
    underlying HTTP client should not impose a lower default limit than the user
    requested. We keep OpenAI's proxy/follow-redirect behavior while using an
    unlimited active connection pool and a conservative keepalive pool.
    """

    return OpenAIAsyncHttpxClient(
        timeout=STREAM_TIMEOUT,
        limits=STREAM_LIMITS,
    )


def _get_file_logger() -> logging.Logger:
    """Get or create file logger for retry events."""
    logger = logging.getLogger("kimi_retry")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / "kimi_retry.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
    return logger


_file_logger = _get_file_logger()


def _log_event(
    event_type: str,
    error_type: str,
    message: str,
    will_retry: bool,
    model_name: str = "",
):
    """Log event to console, file, and inspect-ai transcript."""
    msg_str = str(message)
    status = "RETRY" if will_retry else "ERROR"

    # Try to get span_id from transcript context
    span_id = None
    try:
        t = transcript()
        if t.events:
            span_id = t.events[-1].span_id
    except Exception:
        pass

    # Build context string
    ctx_parts = []
    if model_name:
        ctx_parts.append(f"model={model_name}")
    if span_id:
        ctx_parts.append(f"span={span_id[:8]}")
    ctx_str = f" ({', '.join(ctx_parts)})" if ctx_parts else ""

    log_line = f"[{status}] {error_type}{ctx_str}: {msg_str[:500]}"

    # Console output (truncated)
    print(f"[{status}] {error_type}{ctx_str}: {msg_str[:200]}")

    # File logger (more detail)
    _file_logger.info(log_line)

    # Transcript (saved to .eval file)
    try:
        transcript().info(
            {
                "event_type": event_type,
                "error_type": error_type,
                "message": msg_str,
                "will_retry": will_retry,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
            },
            source="kimi_model",
        )
    except Exception as e:
        print(f"[DEBUG] transcript unavailable: {type(e).__name__}", file=sys.stderr)


class KimiAPI(OpenAICompatibleAPI):
    """Kimi API provider with streaming and custom retry logic."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        stream: bool = False,
        **model_args,
    ) -> None:
        if "http_client" not in model_args:
            model_args["http_client"] = _default_http_client()

        super().__init__(
            model_name=model_name,
            base_url=base_url or os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
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
            if started:
                _log_event("stream_interrupted", type(e).__name__, f"{e} (finish_reason={finish_reason})", False, self.model_name)
                raise
            else:
                _log_event("connection_error", type(e).__name__, str(e), True, self.model_name)
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
            _log_event("length_exceeded", "max_tokens", f"id={completion_id}, usage={usage}", False, self.model_name)

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
        """Retry on rate limit and connection errors."""
        if isinstance(ex, RateLimitError):
            _log_event("retry", "RateLimitError", str(ex), True, self.model_name)
            return True
        if isinstance(ex, APIStatusError) and ex.status_code == 429:
            _log_event("retry", "429", str(ex), True, self.model_name)
            return True
        if isinstance(ex, (APIConnectionError, *RETRYABLE_READ_ERRORS)):
            _log_event("retry", type(ex).__name__, str(ex), True, self.model_name)
            return True

        _log_event("error", type(ex).__name__, str(ex), False, self.model_name)
        return False

    @override
    def retry_wait(self):
        return wait_exponential_jitter(initial=1, max=60, jitter=2)


@modelapi(name="kimi")
def kimi() -> type[KimiAPI]:
    return KimiAPI
