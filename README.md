# Kimi Vendor Verifier

English | [中文](README_zh.md)

A model evaluation tool based on [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) framework for benchmarking Kimi models.

## Supported Benchmarks

| Benchmark | Description | Dataset |
|-----------|-------------|---------|
| **AIME 2025** | American Invitational Mathematics Examination | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) |
| **MMMU Pro Vision** | Multimodal understanding (vision only) | [MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| **OCRBench** | OCR text recognition | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) |

## Setup

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync && uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Configure Environment

```bash
export KIMI_API_KEY="your-api-key"
export KIMI_BASE_URL="your-base-url"
```

Or copy `.env.example` to `.env` and fill in the values.

## Running Evaluations

### Basic Usage

```bash
# Run a single benchmark (required: --model, --temperature, --max-tokens)
uv run python eval.py <benchmark> --model kimi/<model-id> \
    --temperature <temp> --max-tokens <tokens> [options]

# Example: Run AIME 2025
uv run python eval.py aime2025 --model kimi/your-model-id \
    --temperature 0.6 --max-tokens 32768 --stream
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `benchmark` | Task: `aime2025`, `mmmu`, `ocrbench`, `all` | `all` |
| `--model` | Model identifier, e.g., `kimi/your-model-id` | **Required** |
| `--temperature` | Sampling temperature (1.0 for thinking, 0.6 otherwise) | **Required** |
| `--max-tokens` | Max output tokens (see recommended config below) | **Required** |
| `--thinking` | Enable thinking mode (requires `--think-mode kimi/vllm`) | Off |
| `--think-mode` | Thinking param format: `none`, `kimi`, or `vllm` | `none` |
| `--stream` | Enable streaming (recommended for long inference) | Off |
| `--max-connections` | Max concurrent connections | Per benchmark |
| `--epochs` | Number of sampling epochs | Per benchmark |
| `--client-timeout` | HTTP timeout in seconds | `86400` |

> **Thinking Mode Reference**:
>
> | Model Type | Parameters | extra_body |
> |------------|------------|------------|
> | Non-hybrid | Default (no `--think-mode`) | `{}` |
> | Hybrid + thinking off | `--think-mode kimi` | `{"thinking": {"type": "disabled"}}` |
> | Hybrid + thinking on | `--thinking --think-mode kimi` | `{"thinking": {"type": "enabled"}}` |
> | vLLM + thinking off | `--think-mode vllm` | `{"chat_template_kwargs": {"thinking": false}}` |
> | vLLM + thinking on | `--thinking --think-mode vllm` | `{"chat_template_kwargs": {"thinking": true}}` |

### Examples

```bash
# Non-hybrid model (default --think-mode none)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --temperature 0.6 --max-tokens 32768 --stream

# Hybrid model with thinking enabled (Kimi API)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --temperature 1.0 --max-tokens 98304 --stream

# Hybrid model with thinking disabled (Kimi API)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --think-mode kimi --temperature 0.6 --max-tokens 32768 --stream

# Hybrid model with thinking enabled (vLLM/SGLang)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode vllm --temperature 1.0 --max-tokens 98304 --stream
```

## View Results

```bash
# Use inspect view to browse logs
uv run inspect view

# Logs are saved in logs/ directory
```

## Resume Interrupted Evaluations

```bash
uv run inspect eval-retry logs/<log-file>.eval
```

## Project Structure

```
├── eval.py              # Main evaluation CLI
├── kimi_model.py        # Kimi Model API implementation
├── aime2025.py          # AIME 2025 benchmark
├── mmmu_pro_vision.py   # MMMU Pro Vision benchmark
├── ocr_bench.py         # OCRBench benchmark
├── logs/                # Evaluation logs
└── pyproject.toml       # Project configuration
```

## Recommended Configuration

| Benchmark | Temperature | Max Tokens | Epochs |
|-----------|-------------|------------|--------|
| OCRBench | 0.6 / 1.0 | 8192 / 16384 | 1 |
| MMMU | 0.6 / 1.0 | 32768 / 65536 | 1 |
| AIME 2025 | 0.6 / 1.0 | 32768 / 98304 | 32 |

> Format: non-thinking / thinking mode

## Metrics

- **AIME 2025**: `accuracy` / `stderr`, answer format `Answer: <integer>` or `\boxed{<integer>}`
- **MMMU Pro Vision**: `accuracy` / `stderr`, answer format `Answer: $LETTER`
- **OCRBench**: `accuracy` / `stderr`, substring matching

## Notes

### AIME 2025 Evaluation

AIME evaluation generates many output tokens. Keep in mind:

1. **Timeout Settings**
   - **Client**: Default `--client-timeout 86400` (24h), usually no change needed
   - **Server**: Ensure server timeout is also set long enough
   - **Gateway/Proxy**: If using nginx/ALB, adjust `proxy_read_timeout` etc.

2. **Streaming**
   - **Strongly recommended** to use `--stream`
   - Non-streaming requests may timeout in thinking mode
   - Streaming keeps connection alive, avoiding gateway timeouts

3. **Concurrency Control**
   - Default `max_connections=100`, adjust based on server capacity
   - If seeing many 429s or `RemoteProtocolError`, reduce concurrency

4. **Quick Validation**
   - First run with `--epochs 1` to verify configuration
   - Then run full `--epochs 32` evaluation

```bash
# Step 1: Quick validation (30 samples x 1 epoch)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --temperature 1.0 --max-tokens 98304 --stream --epochs 1

# Step 2: Full evaluation (30 samples x 32 epochs)
uv run python eval.py aime2025 --model kimi/your-model-id \
    --thinking --think-mode kimi --temperature 1.0 --max-tokens 98304 --stream
```

### Automatic Retry

The following network errors are **automatically retried** (exponential backoff, 1-60s):

| Error Type | Description |
|------------|-------------|
| `RateLimitError` / `429` | Server rate limiting |
| `APIConnectionError` | Connection failure |
| `ReadError` / `RemoteProtocolError` | Network read error |

> Non-network errors (e.g., model output format issues) are not retried and logged for analysis.

### Stream Interruption Handling

When streaming is interrupted (e.g., `RemoteProtocolError`):
- **Before receiving data**: **Auto retry** (connection failed, safe to retry)
- **After receiving data**: Return partial content, **no retry** (server already processed, retry would cause duplicates)
