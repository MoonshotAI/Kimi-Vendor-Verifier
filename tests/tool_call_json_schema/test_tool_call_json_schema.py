"""Tool-call JSON schema validation tests.

Selected walle-valid MFJS schemas are sent as tool `parameters`, and the
returned `function.arguments` are validated against the original or walle
Canonical schema as appropriate. Both non-streaming and streaming modes are
exercised for the JSONL cases and the supplemental Go-validator cases, which
also cover schemas that must be rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openai
import pytest

from tests.helper import get_thinking_extra_body, token_kwarg
from tests.tool_call_json_schema.validator import (
    REQUEST_MODES,
    ValidatorCase,
    is_object_parameter_schema,
    load_cases,
    select_cases,
    send_tool_schema,
    validate_arguments,
    wrap_schema_as_parameter_property,
)


def _load_selected_cases(config: pytest.Config) -> list[tuple[ValidatorCase, Any, str]]:
    case_dir = Path(config.getoption("case_dir"))
    selection = config.getoption("selection")
    max_cases = config.getoption("max_cases")
    cases = load_cases(case_dir)
    return select_cases(
        cases,
        selection=selection,
        requested_cases=set(),
        max_cases=max_cases,
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_info" not in metafunc.fixturenames:
        return
    selected = _load_selected_cases(metafunc.config)
    params = [
        pytest.param(
            (case, schema, reason, mode),
            id=f"{case.suite}:{case.line}:{mode}",
        )
        for case, schema, reason in selected
        for mode in REQUEST_MODES
    ]
    metafunc.parametrize("case_info", params)


def test_tool_call_schema_matches_case_schema(
    client: openai.Client,
    model: str,
    thinking_flag: bool,
    think_mode: str,
    max_tokens_for_tool_schema: int,
    case_info: tuple[ValidatorCase, Any, str, str],
) -> None:
    """Send a single walle schema as tool parameters and validate arguments."""
    case, schema, selection_reason, mode = case_info
    response = send_tool_schema(
        client,
        model,
        schema,
        max_tokens_for_tool_schema,
        thinking_flag,
        think_mode,
        stream=mode == "stream",
    )

    assert response.accepted, (
        f"{case.suite}:{case.line} [{mode}] ({selection_reason}) "
        f"tool schema rejected: {response.message}"
    )

    valid, message = validate_arguments(schema, response.arguments)
    assert valid, (
        f"{case.suite}:{case.line} [{mode}] ({selection_reason}) "
        f"arguments validation failed: {message}; response: {response.message}"
    )


_WALLE_GO_CASE_PATH = (
    Path(__file__).parents[2]
    / "testdata"
    / "walle_validator_cases"
    / "go_validator_cases.jsonl"
)
_WALLE_GO_CASES: list[dict[str, Any]] = []
_WALLE_GO_LINES = _WALLE_GO_CASE_PATH.read_text(encoding="utf-8").splitlines()
if _WALLE_GO_LINES and _WALLE_GO_LINES[0].startswith(
    "version https://git-lfs.github.com/spec"
):
    raise RuntimeError(
        f"{_WALLE_GO_CASE_PATH} is a Git LFS pointer; "
        "run `git lfs pull` before collecting this suite."
    )
for _line_no, _line in enumerate(_WALLE_GO_LINES, start=1):
    if _line.strip():
        _case = json.loads(_line)
        _case["line"] = _line_no
        _WALLE_GO_CASES.append(_case)
_WALLE_GO_ACCEPTED_CASES = [case for case in _WALLE_GO_CASES if case["accepted"]]
_WALLE_GO_REJECTED_CASES = [case for case in _WALLE_GO_CASES if not case["accepted"]]


@pytest.mark.parametrize(
    "go_case",
    _WALLE_GO_ACCEPTED_CASES,
    ids=lambda go_case: go_case["id"],
)
@pytest.mark.parametrize("mode", REQUEST_MODES)
def test_tool_schema_walle_go_case_matches_canonical(
    client: openai.Client,
    model: str,
    thinking_flag: bool,
    think_mode: str,
    max_tokens_for_tool_schema: int,
    go_case: dict[str, Any],
    mode: str,
) -> None:
    """A walle Go-test schema produces arguments matching its Canonical form."""
    schema = go_case["schema"]
    canonical = go_case["canonical"]
    if not is_object_parameter_schema(schema):
        schema = wrap_schema_as_parameter_property(schema)
        canonical = wrap_schema_as_parameter_property(canonical)
    tool_response = send_tool_schema(
        client,
        model,
        schema,
        max_tokens_for_tool_schema,
        thinking_flag,
        think_mode,
        stream=mode == "stream",
    )
    assert tool_response.accepted, f"{go_case['id']} [{mode}]: {tool_response.message}"
    valid, message = validate_arguments(canonical, tool_response.arguments)
    assert valid, f"{go_case['id']} [{mode}]: {message}"


@pytest.mark.parametrize(
    "go_case",
    _WALLE_GO_REJECTED_CASES,
    ids=lambda go_case: go_case["id"],
)
@pytest.mark.parametrize("mode", REQUEST_MODES)
def test_tool_schema_walle_go_case_rejected(
    client: openai.Client,
    model: str,
    thinking_flag: bool,
    think_mode: str,
    go_case: dict[str, Any],
    mode: str,
) -> None:
    """Walle rejects recursive schemas for which no finite instance exists."""
    with pytest.raises(openai.BadRequestError) as exc_info:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Call the tool."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "walle_reject_case",
                        "parameters": go_case["schema"],
                        "strict": True,
                    },
                }
            ],
            tool_choice="auto",
            extra_body=get_thinking_extra_body(thinking_flag, think_mode, effort="max"),
            stream=mode == "stream",
            **token_kwarg(1),
        )
    assert exc_info.value.status_code == 400
    # Pin the 400 to schema validation, not to an unrelated malformed request.
    assert (
        "schema" in str(exc_info.value).lower()
        or "parameters" in str(exc_info.value).lower()
    )
