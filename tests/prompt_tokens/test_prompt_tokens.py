"""Prompt token groundtruth tests.

Each loaded JSONL case in ``testdata/prompt_token_cases`` contains a chat
completion request and the exact ``expected_prompt_tokens`` the vendor must
report. The test sends every case with ``stream=true`` and compares the
returned ``usage.prompt_tokens`` against the expected constant. Local image
fixtures are resized and encoded as base64 data URLs immediately before the
request.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import openai
from PIL import Image
import pytest


CASE_DIR = Path(__file__).parents[2] / "testdata" / "prompt_token_cases"
CASE_PATHS = (CASE_DIR / "cases.jsonl", CASE_DIR / "vision_cases.jsonl")
IMAGE_FIXTURE_DIR = CASE_DIR / "images"
MAX_IMAGE_DIMENSION = 4096

STANDARD_PARAMS = {
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "seed",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "user",
    "stream_options",
}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"case file not found: {path}")
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            record["_line"] = line_number
            cases.append(record)
    return cases


CASES = [case for path in CASE_PATHS for case in _load_cases(path)]
CASE_IDS = [case["id"] for case in CASES]
if len(CASE_IDS) != len(set(CASE_IDS)):
    duplicates = sorted(
        case_id for case_id in set(CASE_IDS) if CASE_IDS.count(case_id) > 1
    )
    raise ValueError(f"duplicate prompt-token case id(s): {', '.join(duplicates)}")


def _positive_dimension(query: dict[str, list[str]], name: str) -> int:
    values = query.get(name, [])
    if len(values) != 1 or not values[0].isdigit():
        raise ValueError(f"image fixture {name} must be one positive integer")
    value = int(values[0])
    if not 1 <= value <= MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"image fixture {name} must be between 1 and {MAX_IMAGE_DIMENSION}"
        )
    return value


@lru_cache(maxsize=64)
def _image_fixture_data_url(url: str) -> str:
    """Resolve a fixture URL, resize its image, and return a PNG data URL."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "fixture"
        or not parsed.netloc
        or parsed.path
        or parsed.fragment
    ):
        raise ValueError(f"invalid image fixture URL: {url!r}")

    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    if set(query) != {"width", "height"}:
        raise ValueError("image fixture URL requires only width and height")
    width = _positive_dimension(query, "width")
    height = _positive_dimension(query, "height")

    fixture_root = IMAGE_FIXTURE_DIR.resolve()
    fixture_path = (fixture_root / unquote(parsed.netloc)).resolve()
    if not fixture_path.is_relative_to(fixture_root) or not fixture_path.is_file():
        raise ValueError(f"image fixture not found: {parsed.netloc!r}")

    with Image.open(fixture_path) as source:
        if source.format != "PNG":
            raise ValueError(f"image fixture must be a PNG: {parsed.netloc!r}")
        image = source.convert("RGB").resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        )
        buffer = BytesIO()
        image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _materialize_image_fixtures(value: Any) -> Any:
    """Copy a request while replacing local image URLs with base64 data URLs."""
    if isinstance(value, dict):
        result = {
            key: _materialize_image_fixtures(item) for key, item in value.items()
        }
        if result.get("type") == "image_url":
            image_url = result.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url.startswith("fixture://"):
                    image_url["url"] = _image_fixture_data_url(url)
        return result
    if isinstance(value, list):
        return [_materialize_image_fixtures(item) for item in value]
    return value


def _get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _extract_stream_prompt_tokens(stream: Any) -> tuple[int | None, str]:
    """Read usage.prompt_tokens from a streaming response."""
    chunk_count = 0
    for chunk in stream:
        chunk_count += 1
        usages = [
            _get_field(choice, "usage")
            for choice in _get_field(chunk, "choices") or []
        ]
        usages.append(_get_field(chunk, "usage"))
        usage = next((item for item in usages if item is not None), None)
        if usage is None:
            continue
        prompt_tokens = _get_field(usage, "prompt_tokens")
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            return prompt_tokens, f"usage read from stream ({chunk_count} chunk(s))"
        return None, f"stream usage.prompt_tokens is not an integer: {prompt_tokens!r}"
    return None, f"stream ended without usage ({chunk_count} chunk(s))"


def _truncate(value: str, limit: int = 500) -> str:
    value = value.replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _error_message(exc: openai.APIStatusError) -> str:
    parts = [str(exc)]
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            parts.append(response.text)
        except Exception:
            pass
    return "\n".join(part for part in parts if part)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_prompt_tokens_match_groundtruth(
    client: openai.Client,
    model: str,
    case: dict[str, Any],
) -> None:
    """A single prompt-token groundtruth case."""
    request = _materialize_image_fixtures(case["request"])
    request.pop("stream", None)

    extra_body = {
        key: request.pop(key)
        for key in list(request)
        if key not in STANDARD_PARAMS
    }

    kwargs: dict[str, Any] = {
        "model": model,
        "stream": True,
        **request,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        stream = client.chat.completions.create(**kwargs)
    except (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.APIStatusError,
        TimeoutError,
    ) as exc:
        pytest.fail(f"infrastructure/API error for case {case['id']}: {_error_message(exc) if hasattr(exc, 'response') else exc}")

    actual, message = _extract_stream_prompt_tokens(stream)
    expected = case["expected_prompt_tokens"]

    assert actual is not None, (
        f"case {case['id']}: could not read prompt_tokens: {message}"
    )
    assert actual == expected, (
        f"case {case['id']}: prompt_tokens mismatch: "
        f"got {actual}, expected {expected} (diff {actual - expected:+d})"
    )
