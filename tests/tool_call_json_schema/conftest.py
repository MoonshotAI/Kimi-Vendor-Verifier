"""Pytest hooks for the tool_call_json_schema JSON report.

This conftest writes an additional JSON report that mirrors the summary and
result layout previously produced by verify_tool_call_json_schema.py. It is
compatible with pytest-xdist: each worker appends to its own temporary JSONL
file, and the master process merges them in ``pytest_sessionfinish``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest


TOOL_NAME = "kvv_walle_case"
_RESULTS_SUFFIX = ".results.jsonl"
_MAX_MESSAGE_LEN = 2000


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tool-json-report",
        default=os.environ.get("TOOL_JSON_REPORT", "tool-call-schema-report.json"),
        help="Write tool-call schema JSON report to this path "
             "(default: tool-call-schema-report.json)",
    )


def _report_path(config: pytest.Config) -> Path:
    return Path(config.getoption("tool_json_report")).resolve()


_GO_CASE_SUITE = "go_validator_cases"
_GO_CASE_SELECTION_REASON = "go_validator"


def _is_tool_schema_item(item: pytest.Item) -> bool:
    fixturenames = getattr(item, "fixturenames", [])
    return "case_info" in fixturenames or "go_case" in fixturenames


def _record_fields(item: pytest.Item) -> dict[str, Any] | None:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    if "case_info" in callspec.params:
        case, schema, reason, mode = callspec.params["case_info"]
        return {
            "suite": case.suite,
            "line": case.line,
            "mode": mode,
            "selection_reason": reason,
            "schema": schema,
        }
    if "go_case" in callspec.params:
        # Supplemental go_validator_cases.jsonl entries: no selection
        # pipeline; `line` is the JSONL line number, `mode` comes from
        # the test's own parametrization.
        go_case = callspec.params["go_case"]
        return {
            "suite": _GO_CASE_SUITE,
            "line": go_case["line"],
            "mode": callspec.params.get("mode", "non-stream"),
            "selection_reason": _GO_CASE_SELECTION_REASON,
            "schema": go_case["schema"],
            "expected": "accepted" if go_case["accepted"] else "rejected",
        }
    return None


def _worker_results_path(report_path: Path) -> Path:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    return report_path.parent / f"{report_path.name}{_RESULTS_SUFFIX}.{worker}"


def pytest_configure(config: pytest.Config) -> None:
    """Master process cleans up stale per-worker result files."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    report_path = _report_path(config)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for p in report_path.parent.glob(f"{report_path.name}{_RESULTS_SUFFIX}.*"):
        p.unlink(missing_ok=True)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """Record the outcome of every tool-call schema test call."""
    outcome = yield
    if call.when != "call" or not _is_tool_schema_item(item):
        return

    fields = _record_fields(item)
    if fields is None:
        return

    report: pytest.TestReport = outcome.get_result()
    message = ""
    if report.outcome != "passed" and report.longrepr is not None:
        text = str(report.longrepr)
        if len(text) > _MAX_MESSAGE_LEN:
            text = text[: _MAX_MESSAGE_LEN - 3] + "..."
        message = text

    record = {
        **fields,
        "status": report.outcome,
        "message": message,
    }

    path = _worker_results_path(_report_path(item.config))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_worker_results(report_path: Path) -> list[dict[str, Any]]:
    """Read and remove per-worker JSONL result files."""
    results: list[dict[str, Any]] = []
    pattern = f"{report_path.name}{_RESULTS_SUFFIX}.*"
    for p in sorted(report_path.parent.glob(pattern)):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))
        p.unlink(missing_ok=True)
    return results


def _dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last record per (suite, line, mode).

    pytest-rerunfailures may append multiple records for the same item; the
    final one represents the definitive outcome.
    """
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for r in results:
        key = (r["suite"], r["line"], r["mode"])
        by_key[key] = r
    return list(by_key.values())


def _selected_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, str]] = set()
    cases: list[dict[str, Any]] = []
    for r in results:
        key = (r["suite"], r["line"], r["selection_reason"])
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            {
                "suite": r["suite"],
                "line": r["line"],
                "selection_reason": r["selection_reason"],
                "schema": r["schema"],
            }
        )
    return cases


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_mode: dict[str, dict[str, int]] = {}
    for r in results:
        status = r["status"]
        statuses[status] = statuses.get(status, 0) + 1
        reason = r["selection_reason"]
        by_reason[reason] = by_reason.get(reason, 0) + 1
        mode_stats = by_mode.setdefault(r["mode"], {})
        mode_stats[status] = mode_stats.get(status, 0) + 1
    return {
        "total": len(results),
        "by_status": dict(sorted(statuses.items())),
        "by_selection_reason": dict(sorted(by_reason.items())),
        "by_mode": {
            mode: dict(sorted(stats.items()))
            for mode, stats in sorted(by_mode.items())
        },
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Merge per-worker records and write the final JSON report."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    report_path = _report_path(session.config)
    raw_results = _load_worker_results(report_path)
    if not raw_results:
        return

    results = _dedupe_results(raw_results)
    results.sort(key=lambda r: (r["suite"], r["line"], r["mode"]))

    config = session.config
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": config.getoption("smoke_model"),
        "base_url": config.getoption("base_url"),
        "tool_name": TOOL_NAME,
        "dry_run": False,
        "thinking": bool(config.getoption("thinking")),
        "think_mode": config.getoption("think_mode"),
        "modes": sorted({r["mode"] for r in results}),
        "selected_cases": _selected_cases(results),
        "summary": _summarize(results),
        "results": results,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
