"""Pytest fixtures for the migrated k3_features API contract tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest


# Moonshot official API endpoints. Tests that need to distinguish vendor vs.
# official deployments use this list via the `is_vendor` fixture.
MOONSHOT_APIS = [
    "https://api.moonshot.cn/v1",
    "https://api.msh.team/v1",
    "https://api-intl.msh.team/v1",
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--base-url",
        default=os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        help="API base URL (default: KIMI_BASE_URL or https://api.moonshot.cn/v1)",
    )
    parser.addoption(
        "--api-key",
        default=os.environ.get("KIMI_API_KEY", ""),
        help="API key (default: KIMI_API_KEY)",
    )
    parser.addoption(
        "--smoke-model",
        default=os.environ.get("MODEL_NAME", ""),
        help="Model name to test (default: MODEL_NAME)",
    )
    parser.addoption(
        "--think-mode",
        choices=["kimi", "opensource", "none"],
        default=os.environ.get("THINK_MODE", "kimi"),
        help="Thinking parameter format (default: THINK_MODE or kimi)",
    )
    parser.addoption(
        "--case-dir",
        default=os.environ.get(
            "WALLE_CASE_DIR",
            str(Path(__file__).parent.parent / "testdata" / "walle_validator_cases" / "validator_cases"),
        ),
        help="walle validator_cases directory (default: testdata/walle_validator_cases/validator_cases)",
    )
    parser.addoption(
        "--selection",
        choices=["all", "explicit", "object"],
        default=os.environ.get("WALLE_SELECTION", "all"),
        help="Case selection mode: all, explicit, or object (default: all)",
    )
    parser.addoption(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of selected cases to run",
    )
    parser.addoption(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for tool-call schema requests",
    )
    parser.addoption(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("WALLE_MAX_TOKENS", "2048")),
        help="Maximum output tokens for each tool-call response (default: 2048)",
    )


@pytest.fixture(scope="session")
def key(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("api_key")


@pytest.fixture(scope="session")
def think_mode(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("think_mode")


@pytest.fixture(scope="session")
def case_dir(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("case_dir")


@pytest.fixture(scope="session")
def selection(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("selection")


@pytest.fixture(scope="session")
def max_cases(pytestconfig: pytest.Config) -> int | None:
    return pytestconfig.getoption("max_cases")


@pytest.fixture(scope="session")
def thinking_flag(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("thinking"))


@pytest.fixture(scope="session")
def max_tokens_for_tool_schema(pytestconfig: pytest.Config) -> int:
    return pytestconfig.getoption("max_tokens")


@pytest.fixture(scope="function")
def is_vendor(pytestconfig: pytest.Config) -> bool:
    base_url = pytestconfig.getoption("base_url")
    return base_url not in MOONSHOT_APIS


@pytest.fixture(scope="function")
def model(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("smoke_model")


@pytest.fixture(scope="function")
def client(pytestconfig: pytest.Config, key: str) -> openai.Client:
    return openai.Client(
        api_key=key,
        base_url=str(pytestconfig.getoption("base_url")).rstrip("/"),
        timeout=120,
    )


@pytest.fixture(scope="function")
def hclient(pytestconfig: pytest.Config, key: str) -> Any:
    http_client = httpx.Client(
        base_url=str(pytestconfig.getoption("base_url")).rstrip("/"),
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
    )
    yield http_client
    http_client.close()
