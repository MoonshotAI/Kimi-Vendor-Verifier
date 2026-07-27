"""Tool-call JSON schema validation tests.

Each selected walle-valid MFJS schema is sent as a tool's `parameters`, the
model is forced to call the tool, and the returned `function.arguments` are
validated against the schema. Both non-streaming and streaming modes are
exercised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openai
import pytest

from tests.tool_call_json_schema.validator import (
    REQUEST_MODES,
    ValidatorCase,
    load_cases,
    select_cases,
    send_tool_schema,
    validate_arguments,
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
