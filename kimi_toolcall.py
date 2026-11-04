import json

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.model._model import active_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.model._generate_config import active_generate_config
from utils import _copy_request_context, convert_tools, load_dataset


#K3 XTML tiktoken
_XTML_SPECIAL_TOKENS = ["<|open|>", "<|sep|>", "<|close|>", "<|end_of_msg|>"]
_OLD_SPECIAL_TOKENS = ["<|tool_call_end|>"]
_TOTAL_SPECIAL_TOKENS = _XTML_SPECIAL_TOKENS + _OLD_SPECIAL_TOKENS


@solver
def kimi_toolcall_solver() -> Solver:
    """Solver that replays conversation history and calls the model with tools."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tools_raw = state.metadata.get("tools", [])
        tools = convert_tools(tools_raw)

        # This benchmark needs tool schemas in the OpenAI-format request, but
        # must not execute tool calls. Call the active model directly with
        # ToolInfo schemas, then score the returned assistant tool_calls.
        model = active_model()
        if model is None:
            raise RuntimeError("No active model available for kimi_toolcall_solver")

        output = await model.generate(
            input=state.messages,
            tools=tools,
            tool_choice="auto",
            config=active_generate_config(),
        )
        state.output = output
        _copy_request_context(state)
        if output.choices:
            state.messages.append(output.message)
        elif output.completion:
            state.messages.append(ChatMessageAssistant(content=output.completion))
        return state

    return solve


def _is_multi_dict_concat(args: str) -> bool:
    """Detect multiple JSON dict concatenation."""
    if args.count("}{") <= 0:
        return False
    parts = args.split("}{")
    valid = 0
    for i, p in enumerate(parts):
        if i == 0:
            p = p + "}"
        elif i == len(parts) - 1:
            p = "{" + p
        else:
            p = "{" + p + "}"
        try:
            json.loads(p)
            valid += 1
        except json.JSONDecodeError:
            pass
    return valid >= 2


def _has_control_token_leak(args: str) -> str | None:
    for token in _TOTAL_SPECIAL_TOKENS:
        if token in args:
            return token


def _is_invalid_json(args: str) -> bool:
    try:
        json.loads(args)
        return False
    except json.JSONDecodeError:
        return True


_MEANINGLESS_VALUES = ("", None, " ", ": ", ":")


def _is_placeholder_token(value) -> bool:
    """Detect common placeholder tokens like ':' or ':123'."""
    if value in _MEANINGLESS_VALUES:
        return True
    if isinstance(value, str) and value.startswith(":") and value[1:].isdigit():
        return True
    return False


def _is_empty_or_meaningless(args: str) -> bool:
    if not args or args == "{}":
        return True
    try:
        parsed = json.loads(args)
        if isinstance(parsed, dict):
            all_empty = all(_is_placeholder_token(v) for v in parsed.values())
            return all_empty and len(parsed) > 0
    except json.JSONDecodeError:
        pass
    return False


def _is_string_schema(prop: dict) -> bool:
    """Check whether a JSON schema property can be a string value."""
    if not isinstance(prop, dict):
        return False
    if prop.get("type") == "string":
        return True
    if "anyOf" in prop:
        return any(_is_string_schema(p) for p in prop["anyOf"])
    if "oneOf" in prop:
        return any(_is_string_schema(p) for p in prop["oneOf"])
    return False


def _has_explicit_string_constraint(prop: dict) -> bool:
    """Check if schema already constrains the string value (enum, pattern, etc.)."""
    if not isinstance(prop, dict):
        return False
    return any(k in prop for k in ("enum", "pattern", "minLength", "format", "const"))


def _is_semantically_empty(args: str, schema: dict | None) -> tuple[bool, str | None]:
    """Check whether schema-identified string fields have meaningless placeholder values.

    For string fields without explicit schema constraints (enum/pattern/minLength/format),
    reject empty strings or common placeholder tokens like ':'.
    """
    parsed = _parse_arguments(args)
    if not isinstance(parsed, dict):
        return False, None

    if not schema:
        return False, None

    properties = schema.get("properties", {})
    for field_name, prop in properties.items():
        if field_name not in parsed:
            continue
        if not _is_string_schema(prop):
            continue
        if _has_explicit_string_constraint(prop):
            continue

        value = parsed[field_name]
        if not isinstance(value, str):
            continue

        if _is_placeholder_token(value) or value.strip() in {":"}:
            return True, (
                f"{field_name}={value!r} is semantically empty "
                "(schema allows any string, but value is a placeholder)"
            )

    return False, None


def _parse_arguments(args: str):
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return None


def _tool_call_fields(tool_call) -> tuple[str, str, str]:
    """Return id, function name, and raw arguments for dict or ToolCall objects."""
    if isinstance(tool_call, dict):
        func = tool_call.get("function", {})
        args = func.get("arguments", "")
        return (
            str(tool_call.get("id", "")),
            str(func.get("name", "")),
            args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
        )

    call_id = str(getattr(tool_call, "id", ""))
    function = str(getattr(tool_call, "function", ""))
    args = getattr(tool_call, "arguments", "")
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return call_id, function, args


def _tool_schema_by_name(tools: list[dict]) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function", {})
        name = func.get("name")
        parameters = func.get("parameters")
        if isinstance(name, str) and isinstance(parameters, dict):
            schemas[name] = parameters
    return schemas


def _validate_tool_call_schema(args: str, schema: dict | None) -> str | None:
    if schema is None:
        return None

    parsed = _parse_arguments(args)
    if parsed is None:
        return None

    try:
        Draft202012Validator(schema).validate(parsed)
    except ValidationError as e:
        path = ".".join(str(part) for part in e.path)
        location = f" at {path}" if path else ""
        return f"{e.message}{location}"
    except SchemaError as e:
        return f"Invalid tool schema: {e.message}"

    return None


def _check_tool_call_errors(tool_calls: list, tools: list[dict] | None = None) -> list[dict]:
    """Check each tool_call for format and schema errors.
        errors:
        1.control token leak
        2.arguments with multiple dict concat
        3.invalid_json
        4.arguments empty
        5.hallucinate tool
        6.semantically empty (placeholder values in loosely constrained string fields)
        7.xtml special token leak (<|open|>/<|sep|>/<|close|>/<|end_of_msg|> in arguments)
    """
    errors: list[dict] = []
    schemas = _tool_schema_by_name(tools or [])
    for tc in tool_calls:
        call_id, tool_name, args = _tool_call_fields(tc)
        schema = schemas.get(tool_name)
        checks: dict[str, bool] = {}
        details: dict[str, str] = {}

        # Tool must be declared in the request's tools list.
        if not schemas or tool_name not in schemas:
            checks["unknown_tool"] = True
            details["unknown_tool"] = (
                f"tool_name='{tool_name}' is not defined in tools"
            )

        if leaked_token := _has_control_token_leak(args):
            checks["control_token_leak"] = True
            details["control_token_leak"] = (
                f"arguments contain special token {leaked_token}"
            )
        if _is_multi_dict_concat(args):
            checks["multi_dict_concat"] = True
        elif _is_invalid_json(args):
            checks["invalid_json"] = True
        empty_args = not args or args == "{}"
        if _is_empty_or_meaningless(args):
            checks["empty_or_meaningless"] = True

        semantically_empty, sem_error = _is_semantically_empty(args, schema)
        if semantically_empty:
            checks["semantically_empty"] = True
            details["semantically_empty"] = sem_error

        schema_error = _validate_tool_call_schema(args, schema)
        if schema_error:
            checks["schema_validation"] = True
            details["schema_validation"] = schema_error

        if checks:
            error = {
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "arguments": args[:200],
                "checks": checks,
            }
            if details:
                error["details"] = details
            errors.append(error)
    return errors


def _model_output_tool_calls(output) -> list:
    """Extract assistant tool calls from inspect-ai ModelOutput variants."""
    tool_calls: list = []

    for choice in getattr(output, "choices", []) or []:
        msg = getattr(choice, "message", None)
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)

    for msg in getattr(output, "messages", []) or []:
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)

    return tool_calls


def _get_stop_reason(output) -> str | None:
    """Extract stop_reason from the first choice of model output."""
    choices = getattr(output, "choices", []) or []
    if choices:
        return getattr(choices[0], "stop_reason", None)
    return None


@scorer(metrics=[accuracy(), stderr()])
def kimi_toolcall_scorer() -> Scorer:
    """Scorer that checks tool call arguments for format errors."""

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output
        if not output:
            return Score(value=INCORRECT, explanation="No model output")

        stop_reason = _get_stop_reason(output)

        # If the API returned a non-standard finish_reason (e.g. unexpected_state),
        # the model layer records the original value in output metadata. Treat
        # these as incorrect because the model/API did not produce a clean answer.
        original_finish_reason = (
            output.metadata.get("original_finish_reason") if output.metadata else None
        )
        if original_finish_reason is not None:
            return Score(
                value=INCORRECT,
                explanation=f"Model returned non-standard finish_reason: {original_finish_reason}",
            )

        tool_calls = _model_output_tool_calls(output)

        is_repeat_run = bool(state.metadata.get("repeat_run_truncated"))
        target_args_valid = bool(
            state.metadata.get("repeat_run_target_arguments_valid_non_empty")
        )

        if not tool_calls and stop_reason == "stop":
            if is_repeat_run and not target_args_valid:
                return Score(
                    value=CORRECT,
                    explanation=(
                        "No tool_calls in output; repeat-run target is a known bad "
                        "tool-call example, so reproducing it is not required"
                    ),
                )

            # Check if target_assistant_msg had tool_calls.
            target_msg = state.metadata.get("target_assistant_msg")
            if target_msg and target_msg.get("tool_calls"):
                # If every target tool_call is an unknown/undeclared tool, the
                # model is allowed to stop instead of hallucinating it.
                target_tools = state.metadata.get("tools", [])
                target_errors = _check_tool_call_errors(
                    target_msg["tool_calls"], target_tools
                )
                if target_errors and all(
                    "unknown_tool" in e["checks"] for e in target_errors
                ):
                    return Score(
                        value=CORRECT,
                        explanation=(
                            "Model stopped; target tool_calls reference "
                            "undeclared tools"
                        ),
                    )

                return Score(
                    value=INCORRECT,
                    explanation="Model did not produce tool_calls but target had them",
                )
            return Score(
                value=CORRECT,
                explanation="No tool_calls in output (correct if no tools needed)",
            )
        elif not tool_calls and stop_reason != "stop":
            return Score(
                value=INCORRECT,
                explanation=f"Model stopped without tool_calls for non-stop finish_reason: {stop_reason}",
            )

        # Candidate badcase records may forbid exact argument patterns known to
        # trigger failure loops (e.g. '{"command":": null}"}'). These can be
        # syntactically valid JSON, so the format checks above do not catch them.
        forbidden = state.metadata.get("forbidden_arguments") or []
        if forbidden:
            for tc in tool_calls:
                call_id, _, args = _tool_call_fields(tc)
                if args in forbidden:
                    return Score(
                        value=INCORRECT,
                        explanation=(
                            f"Reproduced forbidden arguments pattern "
                            f"(known badcase): tool_call_id={call_id}"
                        ),
                    )

        errors = _check_tool_call_errors(tool_calls, state.metadata.get("tools", []))
        if errors:
            error_types = set()
            for e in errors:
                error_types.update(e["checks"].keys())
            explanation = (
                f"Found {len(errors)} problematic tool_call(s): "
                f"{', '.join(sorted(error_types))}"
            )
            return Score(
                value=INCORRECT,
                answer=json.dumps(errors, ensure_ascii=False),
                explanation=explanation,
            )

        return Score(
            value=CORRECT,
            explanation=f"All {len(tool_calls)} tool_call(s) passed format checks",
        )

    return score


@task
def kimi_toolcall(dataset_path: str, limit: int | None = None, sample_id: str | None = None) -> Task:
    """Kimi Tool Call Regression evaluation task."""
    samples = load_dataset(dataset_path, sample_id=sample_id)
    if limit and limit > 0:
        samples = samples[:limit]

    return Task(
        dataset=MemoryDataset(samples=samples, name="kimi_toolcall"),
        solver=kimi_toolcall_solver(),
        scorer=kimi_toolcall_scorer(),
    )
