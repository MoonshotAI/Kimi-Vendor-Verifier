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
from inspect_ai.model import ChatMessage, GenerateConfig, modelapi
from inspect_ai.model._openai import messages_to_openai as _messages_to_openai
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

# Unlimited read timeout for thinking mode (model may think for a long time)
STREAM_TIMEOUT = httpx.Timeout(timeout=None, connect=60.0)

RETRYABLE_READ_ERRORS = (
    HttpcoreReadError,
    RemoteProtocolError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
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
        certain_provider: str | None = None,
        **model_args,
    ) -> None:
        if certain_provider:
            model_args.setdefault("default_headers", {})
            model_args["default_headers"] = {"X-Msh-Internal-Certain-Provider": certain_provider}
        if "http_client" not in model_args:
            model_args["http_client"] = httpx.AsyncClient(
                timeout=STREAM_TIMEOUT,
                # http2=True,
            )

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
    def emulate_reasoning_history(self) -> bool:
        """Kimi API supports native reasoning_content, no emulation needed."""
        return False

    @override
    async def messages_to_openai(
        self, input: list[ChatMessage]
    ) -> list[dict[str, Any]]:
        """Convert messages to OpenAI format with Kimi reasoning_content support."""
        from inspect_ai.model._openai import openai_chat_message

        def reasoning_handler(reasoning):
            return {"reasoning_content": reasoning.reasoning}

        return [
            await openai_chat_message(msg, reasoning_handler=reasoning_handler)
            for msg in input
        ]

    @override
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[Any],
        tool_choice: Any,
        config: GenerateConfig,
    ) -> Any:
        """Generate and attach original_finish_reason metadata for scorer.
           For keeping self-defined finish_reason.
        """
        result = await super().generate(input, tools, tool_choice, config)

        def attach_metadata(output: Any) -> Any:
            if output is None or isinstance(output, Exception):
                return output
            original = getattr(self, "_last_original_finish_reason", None)
            if original:
                if output.metadata is None:
                    output.metadata = {}
                output.metadata["original_finish_reason"] = original
            return output

        if isinstance(result, tuple):
            output, model_call = result
            return attach_metadata(output), model_call
        return attach_metadata(result)

    @override
    async def _generate_completion(
        self, request: dict[str, Any], config: GenerateConfig
    ) -> ChatCompletion:
        if self.stream:
            request["stream"] = True
            request["stream_options"] = {"include_usage": True}
            return await self._stream_completion(request)
        completion = cast(
            ChatCompletion, await self.client.chat.completions.create(**request)
        )
        return self._normalize_finish_reason(completion)

    def _normalize_finish_reason(self, completion: ChatCompletion) -> ChatCompletion:
        """Normalize non-standard finish_reason and record the original value."""
        if not completion.choices:
            return completion

        valid_finish_reasons = {
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "function_call",
        }
        choice = completion.choices[0]
        original = choice.finish_reason
        if original and original not in valid_finish_reasons:
            tool_calls = bool(choice.message.tool_calls)
            normalized = "tool_calls" if tool_calls else "stop"
            _log_event(
                "finish_reason_normalized",
                original,
                f"mapped to {normalized}",
                False,
                self.model_name,
            )
            # Record original for scorer inspection.
            self._last_original_finish_reason = original
            object.__setattr__(choice, "finish_reason", normalized)
        else:
            self._last_original_finish_reason = None

        return completion

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

        # Normalize non-standard finish_reason values (e.g. "unexpected_state")
        # so inspect-ai's Choice model does not raise a validation error.
        valid_finish_reasons = {"stop", "length", "tool_calls", "content_filter", "function_call"}
        original_finish_reason = finish_reason
        if finish_reason and finish_reason not in valid_finish_reasons:
            normalized = "tool_calls" if tool_calls else "stop"
            _log_event(
                "finish_reason_normalized",
                finish_reason,
                f"mapped to {normalized}",
                False,
                self.model_name,
            )
            finish_reason = normalized
            self._last_original_finish_reason = original_finish_reason
        else:
            self._last_original_finish_reason = None

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


@modelapi(name="opensource")
def opensource() -> type[KimiAPI]:
    return KimiAPI
