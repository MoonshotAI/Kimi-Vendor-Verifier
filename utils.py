import json
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool._tool_call import ToolCall
from inspect_ai.tool._tool_info import ToolInfo


MAX_REPEAT_INPUT_CHARS = 200_000


def _normalize_content(content: Any) -> str:
    """Normalize content to string (handle multimodal list format)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    texts.append("[image]")
                else:
                    texts.append(str(item))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(content)


def convert_messages(openai_messages: list[dict]) -> list:
    """Convert OpenAI-format messages to inspect-ai ChatMessage objects."""
    result: list = []
    for msg in openai_messages:
        role = msg.get("role", "")
        content = _normalize_content(msg.get("content"))
        if role == "system":
            result.append(ChatMessageSystem(content=content))
        elif role == "user":
            result.append(ChatMessageUser(content=content))
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            reasoning_content = msg.get("reasoning_content")

            # Build content with reasoning if present
            # When thinking is enabled, API requires reasoning_content for
            # assistant messages with tool_calls. If missing/empty, add a placeholder.
            assistant_content = content or ""
            if reasoning_content:
                assistant_content = [
                    ContentReasoning(reasoning=reasoning_content),
                    ContentText(text=content or ""),
                ]
            elif tool_calls:
                # thinking mode requires reasoning_content for tool call messages
                assistant_content = [
                    ContentReasoning(reasoning=""),
                    ContentText(text=content or ""),
                ]

            if tool_calls:
                inspect_tool_calls = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    args_raw = func.get("arguments", "")
                    if isinstance(args_raw, dict):
                        args_dict = args_raw
                        parse_err = None
                    elif isinstance(args_raw, str):
                        try:
                            args_dict = json.loads(args_raw) if args_raw else {}
                            parse_err = None
                        except (json.JSONDecodeError, TypeError) as e:
                            args_dict = {}
                            parse_err = str(e)
                    else:
                        args_dict = {}
                        parse_err = f"Unexpected args type: {type(args_raw).__name__}"
                    inspect_tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            function=func_name,
                            arguments=args_dict if isinstance(args_dict, dict) else {},
                            type=tc.get("type", "function"),
                            parse_error=parse_err,
                        )
                    )
                result.append(
                    ChatMessageAssistant(
                        content=assistant_content,
                        tool_calls=inspect_tool_calls,
                    )
                )
            else:
                result.append(ChatMessageAssistant(content=assistant_content))
        elif role == "tool":
            result.append(
                ChatMessageTool(
                    content=content or "",
                    tool_call_id=msg.get("tool_call_id", ""),
                )
            )
        else:
            result.append(ChatMessageUser(content=str(content)))
    return result


def convert_tools(openai_tools: list[dict]) -> list[ToolInfo]:
    """Convert OpenAI-format tool definitions to inspect-ai ToolInfo."""
    tools: list[ToolInfo] = []
    for t in openai_tools:
        func = t.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        tools.append(
            ToolInfo(
                name=name,
                description=description,
                parameters=parameters,
            )
        )
    return tools


def _repeat_run_target_idx(record: dict, messages: list[dict]) -> int | None:
    """Return a valid repeat-run target index, if present."""
    repeat_run = record.get("_repeat_run")
    if not isinstance(repeat_run, dict):
        return None

    excerpts = repeat_run.get("sample_error_excerpts")
    if not excerpts:
        return None
    idx = excerpts[-1].get("msg_idx")
    if not isinstance(idx, int) or idx < 0 or idx >= len(messages):
        return None
    return idx


def _assistant_idx_for_tool_message(messages: list[dict], tool_idx: int) -> int | None:
    """Find the assistant message that produced a tool message."""
    if not (0 <= tool_idx < len(messages)):
        return None
    tool_call_id = messages[tool_idx].get("tool_call_id")
    if not tool_call_id:
        return None

    for idx in range(tool_idx - 1, -1, -1):
        message = messages[idx]
        if message.get("role") != "assistant":
            continue
        tool_call_ids = [
            tool_call.get("id")
            for tool_call in message.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]
        if tool_call_id in tool_call_ids:
            return idx
    return None


def _assistant_tool_call_ids(message: dict) -> list[str]:
    return [
        tool_call.get("id", "")
        for tool_call in message.get("tool_calls") or []
        if isinstance(tool_call, dict) and tool_call.get("id")
    ]


def _repeat_run_input_cut_idx(
    messages: list[dict], target_idx: int
) -> tuple[int | None, bool]:
    """Return a request-safe input cut index and whether the tool group is incomplete."""
    if not isinstance(target_idx, int) or target_idx < 0 or target_idx >= len(messages):
        return None, False
    target_message = messages[target_idx]

    if target_message.get("role") != "tool":
        return target_idx, False

    assistant_idx = _assistant_idx_for_tool_message(messages, target_idx)
    if assistant_idx is None:
        return None, True

    expected_tool_call_ids = set(_assistant_tool_call_ids(messages[assistant_idx]))
    if not expected_tool_call_ids:
        return None, True

    seen_tool_call_ids: set[str] = set()
    for idx in range(assistant_idx + 1, len(messages)):
        message = messages[idx]
        if message.get("role") == "assistant":
            break
        if message.get("role") != "tool":
            continue

        tool_call_id = message.get("tool_call_id")
        if tool_call_id in expected_tool_call_ids:
            seen_tool_call_ids.add(tool_call_id)
        if expected_tool_call_ids.issubset(seen_tool_call_ids):
            return idx + 1, False

    return None, True


def _repeat_run_indices(
    record: dict, messages: list[dict]
) -> tuple[int | None, int | None, bool]:
    """Return input cut index, target index, and whether the tool group is incomplete."""
    target_idx = _repeat_run_target_idx(record, messages)
    if target_idx is None:
        return None, None, False

    input_cut_idx, incomplete_tool_response = _repeat_run_input_cut_idx(
        messages, target_idx
    )
    if input_cut_idx is None:
        return None, target_idx, incomplete_tool_response
    return input_cut_idx, target_idx, incomplete_tool_response


def _tool_call_arguments_valid_non_empty(tool_call: dict) -> bool:
    """Check that function arguments are valid JSON and not empty/meaningless."""
    func = tool_call.get("function", {})
    args = func.get("arguments", "")

    if isinstance(args, dict):
        parsed = args
    elif isinstance(args, str) and args:
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return False
    else:
        return False

    if not isinstance(parsed, dict) or not parsed:
        return False

    meaningless_values = ("", None, " ", ": ", ":", ":158")
    return any(value not in meaningless_values for value in parsed.values())


def _message_arguments_valid_non_empty(message: dict) -> bool:
    """Check whether any tool call has valid, non-empty function arguments."""
    if message.get("role") != "assistant":
        return False
    return any(
        _tool_call_arguments_valid_non_empty(tool_call)
        for tool_call in message.get("tool_calls") or []
    )


def _messages_json_chars(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False))


def _trim_repeat_input_messages(
    messages: list[dict], max_chars: int = MAX_REPEAT_INPUT_CHARS
) -> tuple[list[dict], bool, int, int]:
    """Trim repeat-run input and drop large system prompts.

    This benchmark checks tool-call schema formatting, so the large OpenClaw
    system prompts are not required. Keep the most recent pre-repeat context
    that fits within a conservative char budget.
    """
    original_chars = _messages_json_chars(messages)
    non_system_messages = [
        message for message in messages if message.get("role") != "system"
    ]
    system_removed = len(non_system_messages) != len(messages)

    if _messages_json_chars(non_system_messages) <= max_chars:
        trimmed_chars = _messages_json_chars(non_system_messages)
        return non_system_messages, system_removed, original_chars, trimmed_chars

    selected: list[dict] = []
    selected_chars = 2
    suffix_reversed: list[dict] = []

    for message in reversed(non_system_messages):
        candidate = list(reversed(suffix_reversed + [message]))
        candidate_messages = selected + candidate
        candidate_chars = _messages_json_chars(candidate_messages)
        if candidate_chars > max_chars and suffix_reversed:
            break
        if candidate_chars > max_chars and not suffix_reversed:
            break
        suffix_reversed.append(message)
        selected_chars = candidate_chars

    suffix = list(reversed(suffix_reversed))
    while suffix and suffix[0].get("role") == "tool":
        suffix = suffix[1:]

    trimmed = selected + suffix
    trimmed_chars = _messages_json_chars(trimmed)
    return trimmed, True, original_chars, trimmed_chars


def record_to_sample(record: dict) -> Sample:
    """Convert a raw messages-format record from JSONL to an inspect-ai Sample."""
    repeat_run = record.get("_repeat_run", {})
    repeat_run_first_idx = None
    repeat_run_target_idx = None
    repeat_run_target_arguments_valid_non_empty = False
    repeat_run_input_trimmed = False
    repeat_run_input_chars = None
    repeat_run_truncated = False
    repeat_run_incomplete_tool_response = False

    messages = list(record.get("messages", []))
    target_assistant_msg = None

    # Repeat-run datasets mark a bad tool result as target. Keep input through
    # the complete tool-response group that contains the target.
    (
        repeat_run_first_idx,
        repeat_run_target_idx,
        repeat_run_incomplete_tool_response,
    ) = _repeat_run_indices(record, messages)
    if (
        not repeat_run_incomplete_tool_response
        and repeat_run_first_idx is not None
        and repeat_run_target_idx is not None
    ):
        target_assistant_msg = messages[repeat_run_target_idx]
        repeat_run_target_arguments_valid_non_empty = (
            _message_arguments_valid_non_empty(target_assistant_msg)
        )
        messages = messages[:repeat_run_first_idx]
        repeat_run_input_chars = _messages_json_chars(messages)
        repeat_run_input_trimmed = False
        repeat_run_truncated = True
    elif messages and messages[-1].get("role") == "assistant":
        target_assistant_msg = messages.pop()

    target_str = (
        json.dumps(target_assistant_msg, ensure_ascii=False)
        if target_assistant_msg is not None
        else ""
    )
    tools = record.get("tools", [])

    return Sample(
        input=convert_messages(messages),
        target=target_str,
        id=record.get("id"),
        metadata={
            "dt": record.get("dt"),
            "tools": tools,
            "tools_inspect": [t.model_dump() for t in convert_tools(tools)],
            "repeat_run": repeat_run,
            "repeat_run_first_idx": repeat_run_first_idx,
            "repeat_run_target_idx": repeat_run_target_idx,
            "repeat_run_target_arguments_valid_non_empty": (
                repeat_run_target_arguments_valid_non_empty
            ),
            "repeat_run_input_trimmed": repeat_run_input_trimmed,
            "repeat_run_input_chars": repeat_run_input_chars,
            "repeat_run_truncated": repeat_run_truncated,
            "repeat_run_incomplete_tool_response": repeat_run_incomplete_tool_response,
            "target_assistant_msg": target_assistant_msg,
        },
    )


def _tool_call_sequence_errors(messages: list) -> list[str]:
    pending: list[str] = []
    errors: list[str] = []

    for message in messages:
        role = getattr(message, "role", None)
        if role == "assistant":
            for tool_call in getattr(message, "tool_calls", None) or []:
                tool_call_id = getattr(tool_call, "id", "")
                if tool_call_id:
                    pending.append(tool_call_id)
        elif role == "tool":
            tool_call_id = getattr(message, "tool_call_id", "")
            if tool_call_id in pending:
                pending.remove(tool_call_id)
            else:
                errors.append(f"unexpected_tool:{tool_call_id}")

    errors.extend(f"missing_tool:{tool_call_id}" for tool_call_id in pending)
    return errors


def load_dataset(path: str) -> list[Sample]:
    """Load dataset from JSONL with custom record-to-sample mapping."""
    samples: list[Sample] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample = record_to_sample(record)
            samples.append(sample)
    return samples

