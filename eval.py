import argparse
import json

from inspect_ai import eval

from aime2025 import aime2025
from kimi_toolcall import kimi_toolcall
from mmmu_pro_vision import mmmu_pro_10c
from ocr_bench import ocrbench


import kimi_model  # noqa: F401 - registers kimi model API

BENCHMARKS = {
    "ocrbench": ocrbench,
    "mmmu": mmmu_pro_10c,
    "aime2025": aime2025,
    "kimi_toolcall": kimi_toolcall,
}

# Default configs per benchmark (max_connections, epochs)
BENCH_CONFIGS = {
    "ocrbench": {"max_connections": 50, "epochs": 1},
    "mmmu": {"max_connections": 50, "epochs": 1},
    "aime2025": {"max_connections": 50, "epochs": 32},
    "kimi_toolcall": {"max_connections": 5, "epochs": 5},
}


def get_thinking_extra_body(
    thinking: bool,
    mode: str,
    thinking_effort: str | None = None,
) -> dict:
    """Build extra_body for thinking mode based on backend type.

    Args:
        thinking: Enable thinking mode
        mode: Backend type - "kimi", "opensource"
        thinking_effort: Thinking effort. The recommended parameter is
            reasoning_effort: "max"/"high"/"low" enable thinking at that
            effort and "none" disables thinking. It is sent as a top-level
            field for both "kimi" mode and K3 open-source deployments; the
            legacy `thinking` object / `chat_template_kwargs` are only sent
            for K2 compatibility when no effort is given.

    thinking.effort and reasoning_effort coexist: when an effort is given,
    both are sent (the legacy form for K2-compatible stacks, reasoning_effort
    for K3). "none" disables thinking in both forms.
    """
    if mode == "opensource":
        if thinking_effort:
            # K3 using reasoning_effort.
            return {
                "reasoning_effort": thinking_effort,
            }
        # K2 legacy: chat_template_kwargs.
        return {"chat_template_kwargs": {"thinking": thinking}}
    else:  # kimi
        if thinking_effort == "none":
            # thinking.effort has no "none" value; disabling is expressed
            # with type=disabled in the legacy object.
            thinking_body = {"type": "disabled"}
        elif thinking_effort:
            # An explicit effort implies thinking is enabled in both forms.
            thinking_body = {
                "type": "enabled",
                "keep": "all",
                "effort": thinking_effort,
            }
        else:
            # Compatibility path: K2 legacy thinking object.
            thinking_body = {"type": "enabled" if thinking else "disabled"}
        body: dict = {"thinking": thinking_body}
        if thinking_effort:
            body["reasoning_effort"] = thinking_effort
        return body


def run_eval(
    bench_name: str,
    model: str,
    max_tokens: int | None,
    thinking: bool,
    think_mode: str,
    client_timeout: int,
    stream: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    thinking_effort: str | None = None,
    dataset_path: str | None = None,
    sample_id: str | None = None,
    max_completion_tokens: int | None = None,
    **overrides,
):
    """Run a single benchmark evaluation."""
    task = BENCHMARKS[bench_name]
    config = BENCH_CONFIGS[bench_name]

    max_connections = overrides.get("max_connections", config["max_connections"])
    epochs = overrides.get("epochs", config["epochs"])

    # max_completion_tokens is the wire-equivalent alternative to max_tokens.
    # The CLI enforces that exactly one of them is provided.
    token_param = "max_tokens" if max_tokens is not None else "max_completion_tokens"
    effective_max_tokens = max_tokens if max_tokens is not None else max_completion_tokens

    extra_body = get_thinking_extra_body(thinking, think_mode, thinking_effort)

    print(f"\n{'='*60}")
    print(f"Running: {bench_name} | thinking={thinking} | mode={think_mode}")
    print(f"Model: {model}")
    print(f"{token_param}={effective_max_tokens}, max_connections={max_connections}, epochs={epochs}")
    print(f"temperature={temperature}, top_p={top_p}")
    print(f"thinking_effort={thinking_effort}")
    print(f"stream={stream}, extra_body={extra_body}")
    print(f"{'='*60}\n")

    task_args = {}
    if bench_name == "kimi_toolcall" and dataset_path:
        task_args["dataset_path"] = dataset_path
        if sample_id:
            task_args["sample_id"] = sample_id

    eval(
        [task],
        [model],
        max_tokens=effective_max_tokens,
        max_connections=max_connections,
        epochs=epochs,
        extra_body=extra_body,
        retry_on_error=3,
        continue_on_fail=True,
        fail_on_error=False,
        temperature=temperature,
        top_p=top_p,
        task_args=task_args,
        model_args={
            "stream": stream,
            "max_retries": 0,
            "timeout": client_timeout,
            "use_max_completion_tokens": max_tokens is None,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Kimi Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "bench",
        nargs="?",
        choices=list(BENCHMARKS.keys()),
        default="ocrbench",
        help="Benchmark to run (default: ocrbench)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., kimi/your-model-id)",
    )
    token_group = parser.add_mutually_exclusive_group(required=True)
    token_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens, sent as max_tokens (see README for recommended values per benchmark)",
    )
    token_group.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Equivalent of --max-tokens, but sent as max_completion_tokens on the wire (max_tokens is omitted). Mutually exclusive with --max-tokens.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode (requires --think-mode for hybrid models)",
    )
    parser.add_argument(
        "--think-mode",
        choices=["kimi", "opensource"],
        default="kimi",
        help="Thinking param format: kimi (SaaS API) or opensource (vLLM/SGLang/KTransformers) (default: kimi)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        help="Max concurrent connections",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of sampling epochs",
    )
    parser.add_argument(
        "--client-timeout",
        type=int,
        default=86400,
        help="HTTP request timeout in seconds (default: 86400)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming (keeps connection alive for long inference)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (default: 1.0 for thinking, 0.6 for non-thinking)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "--thinking-effort",
        type=str,
        choices=["none", "low", "high", "max"],
        default=None,
        help=(
            "Thinking effort. When set, both the legacy form (thinking "
            "object for kimi, chat_template_kwargs for opensource) and the "
            "top-level reasoning_effort are sent: low/high/max enable "
            "thinking at that effort, 'none' disables thinking in both "
            "forms (no --thinking flag needed). Omit for K2 legacy behavior."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset path (required for kimi_toolcall benchmark)",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Run only the dataset record with this sample id (kimi_toolcall only)",
    )

    args = parser.parse_args()

    overrides = {}
    if args.max_connections is not None:
        overrides["max_connections"] = args.max_connections
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    run_eval(
        args.bench,
        args.model,
        args.max_tokens,
        args.thinking,
        args.think_mode,
        args.client_timeout,
        args.stream,
        args.temperature,
        args.top_p,
        args.thinking_effort,
        args.dataset,
        args.sample_id,
        max_completion_tokens=args.max_completion_tokens,
        **overrides,
    )


if __name__ == "__main__":
    main()
