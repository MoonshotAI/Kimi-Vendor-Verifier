import argparse

from inspect_ai import eval

from aime2025 import aime2025
from mmmu_pro_vision import mmmu_pro_v
from ocr_bench import ocrbench

import kimi_model  # noqa: F401 - registers kimi model API

BENCHMARKS = {
    "ocrbench": ocrbench,
    "mmmu": mmmu_pro_v,
    "aime2025": aime2025,
}

# Default configs per benchmark (max_connections, epochs)
BENCH_CONFIGS = {
    "ocrbench": {"max_connections": 100, "epochs": 1},
    "mmmu": {"max_connections": 100, "epochs": 1},
    "aime2025": {"max_connections": 100, "epochs": 32},
}


def get_thinking_extra_body(thinking: bool, mode: str) -> dict:
    """Build extra_body for thinking mode based on backend type.

    Args:
        thinking: Enable thinking mode
        mode: Backend type - "kimi", "vllm", or "none" (no thinking param)
    """
    if mode == "none":
        # Non-hybrid model, no thinking param needed
        return {}
    elif mode == "vllm":
        if thinking:
            return {"chat_template_kwargs": {"thinking": True}}
        else:
            return {"chat_template_kwargs": {"thinking": False}}
    else:  # kimi
        return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def run_eval(
    bench_name: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    think_mode: str,
    client_timeout: int,
    stream: bool = False,
    **overrides,
):
    """Run a single benchmark evaluation."""
    task = BENCHMARKS[bench_name]
    config = BENCH_CONFIGS[bench_name]

    max_connections = overrides.get("max_connections", config["max_connections"])
    epochs = overrides.get("epochs", config["epochs"])

    extra_body = get_thinking_extra_body(thinking, think_mode)

    print(f"\n{'='*60}")
    print(f"Running: {bench_name} | thinking={thinking} | mode={think_mode}")
    print(f"Model: {model}")
    print(f"temperature={temperature}, max_tokens={max_tokens}, max_connections={max_connections}, epochs={epochs}")
    print(f"stream={stream}, extra_body={extra_body}")
    print(f"{'='*60}\n")

    eval(
        [task],
        [model],
        temperature=temperature,
        max_tokens=max_tokens,
        max_connections=max_connections,
        epochs=epochs,
        extra_body=extra_body,
        retry_on_error=0,
        continue_on_error=True,
        fail_on_error=False,
        model_args={
            "stream": stream,
            "max_retries": 0,
            "timeout": client_timeout,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Kimi Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Non-hybrid model (no --think-mode)
  uv run python eval.py aime2025 --model kimi/your-model-id \\
      --temperature 0.6 --max-tokens 32768 --stream

  # Hybrid model with thinking enabled (Kimi API)
  uv run python eval.py aime2025 --model kimi/your-model-id \\
      --thinking --think-mode kimi --temperature 1.0 --max-tokens 98304 --stream

  # Hybrid model with thinking disabled (Kimi API)
  uv run python eval.py aime2025 --model kimi/your-model-id \\
      --think-mode kimi --temperature 0.6 --max-tokens 32768 --stream
        """,
    )
    parser.add_argument(
        "bench",
        nargs="?",
        choices=["all", *BENCHMARKS.keys()],
        default="all",
        help="Benchmark to run (default: all)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., kimi/your-model-id)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Sampling temperature (recommended: 1.0 for thinking, 0.6 for non-thinking)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Max output tokens (see README for recommended values per benchmark)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode (requires --think-mode for hybrid models)",
    )
    parser.add_argument(
        "--think-mode",
        choices=["none", "kimi", "vllm"],
        default="none",
        help="Thinking param format: none (non-hybrid), kimi, or vllm (default: none)",
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

    args = parser.parse_args()

    overrides = {}
    if args.max_connections is not None:
        overrides["max_connections"] = args.max_connections
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    benchmarks = BENCHMARKS.keys() if args.bench == "all" else [args.bench]
    for bench_name in benchmarks:
        run_eval(
            bench_name,
            args.model,
            args.temperature,
            args.max_tokens,
            args.thinking,
            args.think_mode,
            args.client_timeout,
            args.stream,
            **overrides,
        )


if __name__ == "__main__":
    main()
