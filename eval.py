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

# Default configs per benchmark: (no-thinking, thinking)
BENCH_CONFIGS = {
    "ocrbench": {
        "temperature": (0.6, 1.0),
        "max_tokens": (8192, 16384),
        "max_connections": 100,
        "epochs": 1,
    },
    "mmmu": {
        "temperature": (0.6, 1.0),
        "max_tokens": (32 * 1024, 64 * 1024),
        "max_connections": 100,
        "epochs": 1,
    },
    "aime2025": {
        "temperature": (0.6, 1.0),
        "max_tokens": (32 * 1024, 96 * 1024),
        "max_connections": 100,
        "epochs": 32,
    },
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
            return {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    else:  # kimi
        return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def run_eval(
    bench_name: str,
    model: str,
    thinking: bool,
    think_mode: str,
    retry: int,
    client_timeout: int,
    stream: bool = False,
    **overrides,
):
    """Run a single benchmark evaluation."""
    task = BENCHMARKS[bench_name]
    config = BENCH_CONFIGS[bench_name]

    idx = 1 if thinking else 0
    temperature = config["temperature"][idx]
    max_tokens = config["max_tokens"][idx]
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
        retry_on_error=retry,
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
  # Run AIME 2025 with thinking mode (hybrid model)
  uv run python eval.py aime2025 --thinking --model kimi/your-model-id

  # Run OCRBench with streaming
  uv run python eval.py ocrbench --model kimi/your-model-id --stream

  # Run with non-hybrid model (no thinking param)
  uv run python eval.py aime2025 --model kimi/your-model-id --think-mode none

  # Run all benchmarks with streaming
  uv run python eval.py all --model kimi/your-model-id --stream
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
        "--thinking",
        action="store_true",
        help="Enable thinking mode",
    )
    parser.add_argument(
        "--think-mode",
        choices=["kimi", "vllm", "none"],
        default="kimi",
        help="Thinking config format: kimi, vllm, or none (default: kimi). Use 'none' for non-hybrid models",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., kimi/your-model-id)",
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
        "--retry",
        type=int,
        default=0,
        help="Retry count on error (default: 0)",
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
    if args.max_connections:
        overrides["max_connections"] = args.max_connections
    if args.epochs:
        overrides["epochs"] = args.epochs

    benchmarks = BENCHMARKS.keys() if args.bench == "all" else [args.bench]
    for bench_name in benchmarks:
        run_eval(
            bench_name,
            args.model,
            args.thinking,
            args.think_mode,
            args.retry,
            args.client_timeout,
            args.stream,
            **overrides,
        )


if __name__ == "__main__":
    main()
