"""Tool Call ID.

"""
from pathlib import Path
import json
from typing import Any
import openai
import pytest
from openai import BadRequestError


CASE_DIR = Path(__file__).parents[2] / "testdata" / "toolcall_reject_cases"
CASE_PATHS = (CASE_DIR / "rejected_case.jsonl",)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"case file not found: {path}")

    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            schema_text = raw_line.strip()
            if not schema_text:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            cases.append(record)
    return cases


CASES = [case for path in CASE_PATHS for case in _load_cases(path)]
CASE_IDS = [case["id"] for case in CASES]
if len(CASE_IDS) != len(set(CASE_IDS)):
    duplicates = sorted(
        case_id for case_id in set(CASE_IDS) if CASE_IDS.count(case_id) > 1
    )
    raise ValueError(f"duplicate prompt-token case id(s): {', '.join(duplicates)}")


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_toolcall_id_without_function_name_rejected(client: openai.Client, model: str, case: dict[str, Any]):
    """toolcall badcases."""
    with pytest.raises(BadRequestError) as e:
        completion = client.chat.completions.create(
            model=model,
            messages=case.get("messages", []),
            tools=case.get("tools", []),
            tool_choice="auto",
            extra_body={"thinking": case["thinking"]} if case.get("thinking") else {}
        )
    _, error, _ = e._excinfo
    assert error.status_code == 400

