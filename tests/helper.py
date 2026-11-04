"""Wire-parameter selection for the output-token cap.

Some vendor endpoints reject `max_tokens` and only accept the equivalent
`max_completion_tokens`. Set the environment variable
``KVV_TOKEN_PARAM=max_completion_tokens`` to switch every request built by
the test suites; the two spellings never appear in the same request.
"""

from __future__ import annotations

import os

TOKEN_PARAM = os.environ.get("KVV_TOKEN_PARAM", "max_tokens")

if TOKEN_PARAM not in ("max_tokens", "max_completion_tokens"):
    raise ValueError(
        f"KVV_TOKEN_PARAM must be 'max_tokens' or 'max_completion_tokens', "
        f"got {TOKEN_PARAM!r}"
    )


def token_kwarg(n: int) -> dict[str, int]:
    """Return ``{TOKEN_PARAM: n}`` for **-splatting into client.create() calls."""
    return {TOKEN_PARAM: n}


def thinking_effort_enabled() -> bool:
    return os.environ.get("ENABLE_THINKING_EFFORT", "true").lower() in ("1", "true", "yes")


def get_thinking_extra_body(
    thinking: bool,
    think_mode: str,
    effort: str | None = None
) -> dict[str, Any]:
    """
    1.thinking false, means non-thinking.
    2.thinking true + effort None, means omit effort parameter use default effort setting.
    3.thinking true + effort, use explicit effort parameter.
    """
    if not thinking_effort_enabled():
        if think_mode == "opensource":
            return {"chat_template_kwargs": {"thinking": thinking}}
        else:
            return {"thinking": {"type": "enabled" if thinking else "disabled"}}
    # Opensource resoning_effort="none" disables thinking;
    # Kimi thinking.type disabled and clear keep, effort fields.
    effort = effort if thinking else "none"
    if think_mode == "opensource":
        return {"reasoning_effort": effort} if effort else {}
    # think-mode: kimi
    if not thinking:
        return {"thinking": {"type": "disabled"}}
    if effort and effort != "none":
        return {"thinking": {"type": "enabled", "keep": "all", "effort": effort}}
    return {"thinking": {"type": "enabled", "keep": "all"}}    
