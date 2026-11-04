import asyncio
import contextvars
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
from openai import APIConnectionError, APIStatusError, BadRequestError, RateLimitError
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
from inspect_ai.model._model_call import ModelCall
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
OPENAI_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "function_call"}

# Per-coroutine storage for the original (pre-normalization) finish_reason.
# This avoids race conditions when the same KimiAPI instance is shared across
# concurrent `generate()` calls (e.g. --max-connections 50).
_original_finish_reason_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "original_finish_reason", default=None
)


class _ResponseHeaderTracker:
    """Capture upstream request_id / trace_id from httpx response headers.

    inspect-ai injects a local request id via the ``x-irid`` header.  We use
    that value to correlate the response headers we see in the event hook with
    the ``ModelCall`` returned by ``OpenAICompatibleAPI.generate``.
    """

    _CLIENT_REQUEST_ID_HEADER = "x-irid"
    _TRACE_ID_HEADER = "x-msh-trace-id"
    _REQUEST_ID_HEADERS = (
        "msh-request-id",
        "x-request-id",
        "x-moongate-request-id",
    )
    _MAX_CONTEXTS = 10_000

    def __init__(self) -> None:
        self._contexts: dict[str, dict[str, str | None]] = {}
        self._lock = asyncio.Lock()

    def install(self, client: httpx.AsyncClient) -> None:
        client.event_hooks.setdefault("response", []).append(self._on_response)

    def context_from_response(
        self,
        response: httpx.Response,
        client_request_id: str | None = None,
    ) -> dict[str, str | None] | None:
        client_request_id = client_request_id or response.request.headers.get(
            self._CLIENT_REQUEST_ID_HEADER
        )
        if not client_request_id:
            return None

        request_id: str | None = None
        for header in self._REQUEST_ID_HEADERS:
            value = response.headers.get(header)
            if value:
                request_id = value
                break

        trace_id = response.headers.get(self._TRACE_ID_HEADER)
        return {
            "client_request_id": client_request_id,
            "request_id": request_id,
            "trace_id": trace_id,
        }

    async def _on_response(self, response: httpx.Response) -> None:
        context = self.context_from_response(response)
        if context is None:
            return
        client_request_id = context["client_request_id"]

        async with self._lock:
            if (
                client_request_id not in self._contexts
                and len(self._contexts) >= self._MAX_CONTEXTS
            ):
                # Keep only the newest contexts; these are observability hints.
                self._contexts.pop(next(iter(self._contexts)))
            self._contexts[client_request_id] = context

    async def pop(self, client_request_id: str | None) -> dict[str, str | None] | None:
        if not client_request_id:
            return None
        async with self._lock:
            return self._contexts.pop(client_request_id, None)

    async def peek(self, client_request_id: str | None) -> dict[str, str | None] | None:
        """Read a context without removing it.

        Used for retryable errors (e.g. 429): the attempt may be retried, so
        the entry's lifecycle is left to the success path / eviction instead
        of being consumed here.
        """
        if not client_request_id:
            return None
        async with self._lock:
            return self._contexts.get(client_request_id)


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


def _request_context_log_parts(context: dict[str, str | None] | None) -> list[str]:
    if not context:
        return []

    parts: list[str] = []
    for key in ("request_id", "trace_id", "client_request_id"):
        value = context.get(key)
        if value:
            parts.append(f"{key}={value}")
    return parts


def _log_event(
    event_type: str,
    error_type: str,
    message: str,
    will_retry: bool,
    model_name: str = "",
    request_context: dict[str, str | None] | None = None,
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
    if request_context:
        ctx_parts.extend(_request_context_log_parts(request_context))
    ctx_str = f" ({', '.join(ctx_parts)})" if ctx_parts else ""
    log_line = f"[{status}] {error_type}{ctx_str}: {msg_str[:500]}"

    # Console output (truncated)
    print(f"[{status}] {error_type}{ctx_str}: {msg_str[:200]}")

    # File logger (more detail)
    _file_logger.info(log_line)

    # Transcript (saved to .eval file)
    try:
        payload = {
            "event_type": event_type,
            "error_type": error_type,
            "message": msg_str[:2000],
            "will_retry": will_retry,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
        }
        if request_context is not None:
            payload["request_context"] = request_context
        transcript().info(payload, source="kimi_model")
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
        use_max_completion_tokens: bool = False,
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

        self._response_tracker = _ResponseHeaderTracker()
        self._use_max_completion_tokens = use_max_completion_tokens

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
    def initialize(self) -> None:
        super().initialize()
        self._response_tracker.install(self.client._client)

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
    def completion_params(self, config: GenerateConfig, tools: bool) -> dict[str, Any]:
        params = super().completion_params(config, tools)
        if self._use_max_completion_tokens and "max_tokens" in params:
            # Some vendor endpoints reject max_tokens; send the equivalent
            # max_completion_tokens instead. Never send both.
            params["max_completion_tokens"] = params.pop("max_tokens")
        return params

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
        # Reset per-call state so error paths that never reach
        # _normalize_finish_reason (e.g. a soft-error 400 handled into a
        # ModelOutput) cannot inherit a stale value from a previous
        # generate() in the same task context.
        _original_finish_reason_var.set(None)
        result = await super().generate(input, tools, tool_choice, config)

        def attach_metadata(output: Any) -> Any:
            if output is None or isinstance(output, Exception):
                return output
            original = _original_finish_reason_var.get()
            if original:
                if output.metadata is None:
                    output.metadata = {}
                output.metadata["original_finish_reason"] = original
            return output

        if isinstance(result, tuple):
            output, model_call = result
            output = attach_metadata(output)
            await self._attach_request_context(output, model_call)
            return output, model_call
        return attach_metadata(result)

    async def _attach_request_context(
        self,
        output: Any,
        model_call: ModelCall | None,
    ) -> None:
        """Write trace_id / request_id into ModelOutput metadata and ModelCall."""
        if output is None or isinstance(output, Exception) or model_call is None:
            return

        request = model_call.request or {}
        extra_headers = request.get("extra_headers") or {}
        client_request_id = extra_headers.get("x-irid")
        if not client_request_id:
            return

        context = await self._response_tracker.pop(client_request_id)
        if context is None:
            context = {
                "client_request_id": client_request_id,
                "request_id": None,
                "trace_id": None,
            }

        if output.metadata is None:
            output.metadata = {}
        output.metadata.setdefault("request_context", {}).update(context)

        if model_call.response is not None:
            model_call.response["_request_context"] = context

    async def _capture_error_request_context(
        self,
        request: dict[str, Any],
        ex: APIStatusError,
    ) -> dict[str, str | None] | None:
        """Attach captured request context to API errors for sync retry logging.

        Prefers the response carried by the exception and falls back to the
        response-header tracker. The tracker entry is only peeked here, never
        consumed: ``_attach_request_context`` is the single pop site, so a
        soft-error 400 (e.g. context_length_exceeded, which still produces a
        ModelOutput) can record the real context in output metadata. 400
        errors are terminal (never retried), so they are logged here and
        flagged to keep ``should_retry`` from logging a duplicate; logging of
        other statuses (e.g. 429) is left to the ``should_retry`` branches.
        """
        is_terminal_400 = getattr(ex, "status_code", None) == 400

        extra_headers = request.get("extra_headers") or {}
        client_request_id = extra_headers.get("x-irid")
        if not client_request_id:
            response = getattr(ex, "response", None)
            if response is not None:
                client_request_id = response.request.headers.get("x-irid")
        if not client_request_id:
            return None

        response = getattr(ex, "response", None)
        context = None
        if response is not None:
            context = self._response_tracker.context_from_response(
                response,
                client_request_id=client_request_id,
            )

        tracked_context = await self._response_tracker.peek(client_request_id)
        if context is None:
            context = tracked_context
        elif tracked_context is not None:
            for key in ("request_id", "trace_id"):
                if not context.get(key) and tracked_context.get(key):
                    context[key] = tracked_context[key]

        if context is None:
            context = {
                "client_request_id": client_request_id,
                "request_id": None,
                "trace_id": None,
            }
        setattr(ex, "_kimi_request_context", context)
        if is_terminal_400:
            setattr(ex, "_kimi_request_context_logged", True)
            _log_event(
                "error",
                type(ex).__name__,
                str(ex),
                False,
                self.model_name,
                request_context=context,
            )
        return context

    @override
    async def _generate_completion(
        self, request: dict[str, Any], config: GenerateConfig
    ) -> ChatCompletion:
        try:
            if self.stream:
                request["stream"] = True
                request["stream_options"] = {"include_usage": True}
                return await self._stream_completion(request)
            completion = cast(
                ChatCompletion, await self.client.chat.completions.create(**request)
            )
            return self._normalize_finish_reason(completion)
        except (BadRequestError, APIStatusError) as ex:
            await self._capture_error_request_context(request, ex)
            raise

    def _normalize_finish_reason(self, completion: ChatCompletion) -> ChatCompletion:
        """Normalize non-standard finish_reason and record the original value."""
        _original_finish_reason_var.set(None)
        if not completion.choices:
            return completion

        choice = completion.choices[0]
        original = choice.finish_reason
        if original and original not in OPENAI_FINISH_REASONS:
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
            _original_finish_reason_var.set(original)
            object.__setattr__(choice, "finish_reason", normalized)

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
        original_finish_reason = finish_reason
        if finish_reason and finish_reason not in OPENAI_FINISH_REASONS:
            normalized = "tool_calls" if tool_calls else "stop"
            _log_event(
                "finish_reason_normalized",
                finish_reason,
                f"mapped to {normalized}",
                False,
                self.model_name,
            )
            finish_reason = normalized
            _original_finish_reason_var.set(original_finish_reason)
        elif not finish_reason:
            _log_event("finish_reason_missing", "stream_end", f"id={completion_id}, usage={usage}", False, self.model_name)
            _original_finish_reason_var.set("missing")
        else:
            _original_finish_reason_var.set(None)

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
        request_context = getattr(ex, "_kimi_request_context", None)
        if isinstance(ex, RateLimitError):
            _log_event(
                "retry",
                "RateLimitError",
                str(ex),
                True,
                self.model_name,
                request_context=request_context,
            )
            return True
        if isinstance(ex, APIStatusError) and ex.status_code == 429:
            _log_event(
                "retry",
                "429",
                str(ex),
                True,
                self.model_name,
                request_context=request_context,
            )
            return True
        if isinstance(ex, (APIConnectionError, *RETRYABLE_READ_ERRORS)):
            _log_event(
                "retry",
                type(ex).__name__,
                str(ex),
                True,
                self.model_name,
                request_context=request_context,
            )
            return True

        if isinstance(ex, APIStatusError) and ex.status_code == 400:
            if not getattr(ex, "_kimi_request_context_logged", False):
                _log_event(
                    "error",
                    type(ex).__name__,
                    str(ex),
                    False,
                    self.model_name,
                    request_context=request_context,
                )
            return False

        _log_event(
            "error",
            type(ex).__name__,
            str(ex),
            False,
            self.model_name,
            request_context=request_context,
        )
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
