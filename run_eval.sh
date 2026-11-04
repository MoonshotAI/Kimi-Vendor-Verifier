#!/usr/bin/env bash
# Run individual evaluation jobs locally, mirroring .gitlab-ci.yml.
#
# Usage:
#   ./run_eval.sh <job_name>
#   ./run_eval.sh all              # run all jobs applicable to current settings
#   ./run_eval.sh list             # list available jobs
#
# Required environment variables (pick one naming convention):
#   KIMI_API_KEY or API_KEY
#   KIMI_BASE_URL or BASE_URL
#   MODEL_NAME
#
# Optional:
#   THINK_MODE            default: kimi
#   ENABLE_THINKING_EFFORT default: true
#   BEAM_PLAIN_TEXT       default: true; "false" disables --plain-text for beam_generate
#   JUDGE_BASE_URL        for beam_judge (required by "all" when ENABLE_THINKING_EFFORT=true)
#   JUDGE_API_KEY         for beam_judge (required by "all" when ENABLE_THINKING_EFFORT=true)
#   USE_PROXY             set to "1" to enable the CI proxy setup

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment normalization
# ---------------------------------------------------------------------------
export KIMI_API_KEY="${KIMI_API_KEY:-${API_KEY:-}}"
export KIMI_BASE_URL="${KIMI_BASE_URL:-${BASE_URL:-}}"
export THINK_MODE="${THINK_MODE:-kimi}"
export ENABLE_THINKING_EFFORT="${ENABLE_THINKING_EFFORT:-true}"
export BEAM_PLAIN_TEXT="${BEAM_PLAIN_TEXT:-true}"
export INSPECT_LOG_LEVEL="ERROR"


require_env() {
    local missing=0
    [ -z "${KIMI_API_KEY:-}" ]  && { echo "ERROR: KIMI_API_KEY or API_KEY must be set" >&2; missing=1; }
    [ -z "${KIMI_BASE_URL:-}" ] && { echo "ERROR: KIMI_BASE_URL or BASE_URL must be set" >&2; missing=1; }
    [ -z "${MODEL_NAME:-}" ]    && { echo "ERROR: MODEL_NAME must be set" >&2; missing=1; }
    if [ "$missing" = "1" ]; then
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Optional proxy setup (same as CI)
# ---------------------------------------------------------------------------
setup_proxy() {
    if [ "${USE_PROXY:-0}" != "1" ]; then
        return
    fi

    echo "=== Proxy env check ==="
    echo "HTTP_PROXY=$HTTP_PROXY"
    echo "HTTPS_PROXY=$HTTPS_PROXY"
    echo "NO_PROXY=$NO_PROXY"
    echo "======================="
}

# ---------------------------------------------------------------------------
# Dependency sync (mirrors CI before_script)
# ---------------------------------------------------------------------------
uv_sync() {
    uv sync && uv pip install -e .
}

# ---------------------------------------------------------------------------
# Job definitions (matching .gitlab-ci.yml)
# ---------------------------------------------------------------------------

job_cache_warmup() {
    echo "=== cache-warmup ==="
    uv run python -c "
from datasets import load_dataset
load_dataset('echo840/OCRBench', split='test')
load_dataset('MMMU/MMMU_Pro', 'standard (10 options)', split='test')
load_dataset('math-ai/aime25', split='test')
print('Cache warmup complete!')
"
}

job_verify_parameters() {
    echo "=== verify_parameters ==="
    local skip_hosts="api\.msh\.team|api\.moonshot\.cn|api-intl\.msh\.team|api\.moonshot\.ai"
    if echo "${KIMI_BASE_URL}" | grep -Eq "${skip_hosts}"; then
        echo "Skipping verify_parameters for official Moonshot API: ${KIMI_BASE_URL}"
        return 0
    fi
    uv run pytest -ra tests/params \
        --base-url "${KIMI_BASE_URL}" \
        --api-key "${KIMI_API_KEY}" \
        --smoke-model "${MODEL_NAME}" \
        --think-mode "$THINK_MODE" \
        --junitxml=params-report.xml
}

job_verify_tool_call_json_schema() {
    echo "=== verify_tool_call_json_schema ==="
    uv run pytest -n 4 tests/tool_call_json_schema \
        --base-url "${KIMI_BASE_URL}" \
        --api-key "${KIMI_API_KEY}" \
        --smoke-model "${MODEL_NAME}" \
        --think-mode "$THINK_MODE" \
        --thinking \
        --junitxml=tool-call-schema-report.xml \
        --tool-json-report=tool-call-schema-report.json \
        --reruns 3 \
        --reruns-delay 2 \
        --rerun-show-tracebacks \
        -ra -v
}

job_verify_toolcall_result_reject() {
    echo "=== verify_toolcall_reject ==="
    uv run pytest -ra -v tests/toolcall_reject_cases/ \
        --base-url "${KIMI_BASE_URL}" \
        --api-key "${KIMI_API_KEY}" \
        --smoke-model "${MODEL_NAME}" \
        --junitxml=toolcall-reject-report.xml
}

job_verify_prompt_tokens() {
    echo "=== verify_prompt_tokens ==="
    uv run pytest -ra -v tests/prompt_tokens \
        --base-url "${KIMI_BASE_URL}" \
        --api-key "${KIMI_API_KEY}" \
        --smoke-model "${MODEL_NAME}" \
        --junitxml=prompt-tokens-report.xml
}

job_verify_k3_features() {
    echo "=== verify_k3_features ==="
    uv run pytest -n 5 -ra -v tests/k3_features \
        --base-url "${KIMI_BASE_URL}" \
        --api-key "${KIMI_API_KEY}" \
        --smoke-model "${MODEL_NAME}" \
        --junitxml=k3-features-report.xml
}

job_beam_generate() {
    echo "=== beam_generate ==="
    (
        cd beam
        export BEAM_API_KEY="${KIMI_API_KEY}"
        PLAIN_TEXT_ARG=""
        if [ "$BEAM_PLAIN_TEXT" = "true" ]; then PLAIN_TEXT_ARG="--plain-text"; fi
        uv run python beam_generate.py \
            --model "${MODEL_NAME}" \
            --base-url "${KIMI_BASE_URL}" \
            --concurrency 8 \
            --output answers.jsonl \
            $PLAIN_TEXT_ARG
    )
}

job_beam_judge() {
    echo "=== beam_judge ==="
    if [ -z "${JUDGE_BASE_URL:-}" ] || [ -z "${JUDGE_API_KEY:-}" ]; then
        echo "ERROR: JUDGE_BASE_URL and JUDGE_API_KEY must be set for beam_judge" >&2
        exit 1
    fi
    cd beam
    uv run python beam_judge.py \
        --answers answers.jsonl \
        --judge-model kimi-k2.6 \
        --judge-base-url "${JUDGE_BASE_URL}" \
        --judge-api-key "${JUDGE_API_KEY}" \
        --judge-temperature 1.0 \
        --concurrency 10 \
        --output scores.jsonl
    cd -
}

job_ocrbench_non_thinking() {
    echo "=== ocrbench_non_thinking ==="
    uv run python eval.py ocrbench \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 8192 \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 50 \
        --temperature 0.6 \
        --top-p 0.95
}

job_ocrbench_thinking() {
    echo "=== ocrbench_thinking ==="
    uv run python eval.py ocrbench \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 16384 \
        --thinking \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 50 \
        --temperature 1.0 \
        --top-p 0.95
}

job_ocrbench_thinking_effort() {
    local effort="${1:-max}"
    echo "=== ocrbench_thinking_effort (effort=${effort}) ==="
    uv run python eval.py ocrbench \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 16384 \
        --thinking \
        --thinking-effort "$effort" \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 50 \
        --temperature 1.0 \
        --top-p 0.95
}

job_MMMU_pro_non_thinking() {
    echo "=== MMMU_pro_non_thinking ==="
    uv run python eval.py mmmu \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 65536 \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 50 \
        --temperature 0.6 \
        --top-p 0.95
}

job_MMMU_pro_thinking() {
    echo "=== MMMU_pro_thinking ==="
    uv run python eval.py mmmu \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 65536 \
        --thinking \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 50 \
        --temperature 1.0 \
        --top-p 0.95
}

job_MMMU_pro_thinking_effort() {
    local effort="${1:-max}"
    echo "=== MMMU_pro_thinking_effort (effort=${effort}) ==="
    uv run python eval.py mmmu \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 98304 \
        --thinking \
        --thinking-effort "$effort" \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 50 \
        --temperature 1.0 \
        --top-p 0.95
}

job_AIME_non_thinking() {
    echo "=== AIME_non_thinking ==="
    uv run python eval.py aime2025 \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 16384 \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 100 \
        --temperature 0.6 \
        --top-p 0.95
}

job_AIME_thinking() {
    echo "=== AIME_thinking ==="
    uv run python eval.py aime2025 \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 98304 \
        --thinking \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 50 \
        --temperature 1.0 \
        --top-p 0.95
}

job_toolcall_thinking() {
    echo "=== toolcall_thinking ==="
    uv run python eval.py kimi_toolcall \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 32768 \
        --thinking \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 50 \
        --temperature 1.0 \
        --top-p 0.95 \
        --dataset toolcall_benchmark/toolcall_thinking_samples.stable2.jsonl
}

job_toolcall_thinking_effort() {
    local effort="${1:-max}"
    echo "=== toolcall_thinking_effort (effort=${effort}) ==="
    uv run python eval.py kimi_toolcall \
        --model "kimi/${MODEL_NAME}" \
        --max-tokens 32768 \
        --thinking \
        --thinking-effort "$effort" \
        --think-mode "$THINK_MODE" \
        --stream \
        --max-connections 50 \
        --temperature 1.0 \
        --top-p 0.95 \
        --dataset toolcall_benchmark/toolcall_thinking_samples.stable2.jsonl
}

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

list_jobs() {
    echo "Available jobs:"
    echo "  cache-warmup          (optional locally: eval jobs download datasets on first use)"
    echo "  verify_parameters"
    echo "  verify_toolcall_result_reject"
    echo "  verify_tool_call_json_schema"
    echo "  verify_prompt_tokens  (requires ENABLE_THINKING_EFFORT=true)"
    echo "  verify_k3_features    (requires ENABLE_THINKING_EFFORT=true)"
    echo "  beam_generate         (requires ENABLE_THINKING_EFFORT=true)"
    echo "  beam_judge            (requires ENABLE_THINKING_EFFORT=true + judge env)"
    echo "  ocrbench_non_thinking (requires ENABLE_THINKING_EFFORT!=true)"
    echo "  ocrbench_thinking     (requires ENABLE_THINKING_EFFORT!=true)"
    echo "  ocrbench_thinking_effort:<low|high|max>"
    echo "  MMMU_pro_non-thinking (requires ENABLE_THINKING_EFFORT!=true)"
    echo "  MMMU_pro_thinking     (requires ENABLE_THINKING_EFFORT!=true)"
    echo "  MMMU_pro_thinking_effort:<low|high|max>"
    echo "  AIME_non_thinking     (requires ENABLE_THINKING_EFFORT!=true)"
    echo "  AIME_thinking         (requires ENABLE_THINKING_EFFORT!=true)"
    echo "  toolcall_thinking     (requires ENABLE_THINKING_EFFORT!=true)"
    echo "  toolcall_thinking_effort:<low|high|max>"
    echo ""
    echo "Run all applicable jobs:"
    echo "  ./run_eval.sh all"
}

# Pre-flight check for "all": beam_generate / beam_judge only run when
# ENABLE_THINKING_EFFORT=true, so their credentials are only required then.
# Base variables are covered by require_env before this runs.
check_run_all_env() {
    local missing=()

    if [ "$ENABLE_THINKING_EFFORT" = "true" ]; then
        [ -z "${JUDGE_BASE_URL:-}" ] && missing+=("JUDGE_BASE_URL (beam_judge)")
        [ -z "${JUDGE_API_KEY:-}" ] && missing+=("JUDGE_API_KEY (beam_judge)")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: missing required environment variables for 'all':" >&2
        local var
        for var in "${missing[@]}"; do
            echo "  - $var" >&2
        done
        exit 1
    fi
}

# Run a job function with start/finish banners.
run_job() {
    local name="$1"; shift
    echo ""
    echo "==================================================================="
    echo "[START] ${name} ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "==================================================================="
    if "$@"; then
        echo "[OK] ${name}"
    else
        local rc=$?
        echo "[FAIL] ${name} (exit code: ${rc})" >&2
        return "$rc"
    fi
}

run_all() {
    require_env
    check_run_all_env
    setup_proxy
    uv_sync

    # No cache-warmup here: unlike CI, the local HF cache persists between
    # runs, and each eval job downloads its dataset on first use by itself.
    # Run `./run_eval.sh cache-warmup` manually if you want to pre-download.
    run_job "verify_parameters" job_verify_parameters
    run_job "verify_tool_call_json_schema" job_verify_tool_call_json_schema
    run_job "verify_toolcall_result_reject" job_verify_toolcall_result_reject

    if [ "$ENABLE_THINKING_EFFORT" = "true" ]; then
        run_job "verify_prompt_tokens" job_verify_prompt_tokens
        run_job "verify_k3_features" job_verify_k3_features
        run_job "beam_generate" job_beam_generate
        run_job "beam_judge" job_beam_judge

        for effort in low high max; do
            run_job "ocrbench_thinking_effort:${effort}" job_ocrbench_thinking_effort "$effort"
        done
        for effort in low high max; do
            run_job "MMMU_pro_thinking_effort:${effort}" job_MMMU_pro_thinking_effort "$effort"
        done
        for effort in low high max; do
            run_job "toolcall_thinking_effort:${effort}" job_toolcall_thinking_effort "$effort"
        done
    else
        run_job "ocrbench_non_thinking" job_ocrbench_non_thinking
        run_job "ocrbench_thinking" job_ocrbench_thinking
        run_job "MMMU_pro_non_thinking" job_MMMU_pro_non_thinking
        run_job "MMMU_pro_thinking" job_MMMU_pro_thinking
        run_job "AIME_thinking" job_AIME_thinking
        run_job "AIME_non_thinking" job_AIME_non_thinking
        run_job "toolcall_thinking" job_toolcall_thinking
    fi

    echo ""
    echo "==================================================================="
    echo "[DONE] all jobs completed successfully"
    echo "==================================================================="
}

main() {
    local cmd="${1:-list}"

    case "$cmd" in
        list|--help|-h)
            list_jobs
            exit 0
            ;;
        all)
            run_all
            ;;
        cache-warmup)
            require_env; setup_proxy; uv_sync; run_job "cache-warmup" job_cache_warmup ;;
        verify_parameters)
            require_env; setup_proxy; uv_sync; run_job "verify_parameters" job_verify_parameters ;;
        verify_tool_call_json_schema)
            require_env; setup_proxy; uv_sync; run_job "verify_tool_call_json_schema" job_verify_tool_call_json_schema ;;
        verify_prompt_tokens)
            require_env; setup_proxy; uv_sync; run_job "verify_prompt_tokens" job_verify_prompt_tokens ;;
        verify_toolcall_result_reject)
            require_env; setup_proxy; uv_sync; run_job "verify_toolcall_result_reject" job_verify_toolcall_result_reject ;;
        verify_k3_features)
            require_env; setup_proxy; uv_sync; run_job "verify_k3_features" job_verify_k3_features ;;
        beam_generate)
            require_env; setup_proxy; uv_sync; run_job "beam_generate" job_beam_generate ;;
        beam_judge)
            require_env; setup_proxy; uv_sync; run_job "beam_judge" job_beam_judge ;;
        ocrbench_non_thinking)
            require_env; setup_proxy; uv_sync; run_job "ocrbench_non_thinking" job_ocrbench_non_thinking ;;
        ocrbench_thinking)
            require_env; setup_proxy; uv_sync; run_job "ocrbench_thinking" job_ocrbench_thinking ;;
        ocrbench_thinking_effort)
            require_env; setup_proxy; uv_sync; run_job "ocrbench_thinking_effort:${2:-max}" job_ocrbench_thinking_effort "${2:-max}" ;;
        ocrbench_thinking_effort:*)
            require_env; setup_proxy; uv_sync; run_job "$cmd" job_ocrbench_thinking_effort "${cmd#*:}" ;;
        MMMU_pro_non_thinking|MMMU_pro_non-thinking)
            require_env; setup_proxy; uv_sync; run_job "MMMU_pro_non_thinking" job_MMMU_pro_non_thinking ;;
        MMMU_pro_thinking)
            require_env; setup_proxy; uv_sync; run_job "MMMU_pro_thinking" job_MMMU_pro_thinking ;;
        MMMU_pro_thinking_effort)
            require_env; setup_proxy; uv_sync; run_job "MMMU_pro_thinking_effort:${2:-max}" job_MMMU_pro_thinking_effort "${2:-max}" ;;
        MMMU_pro_thinking_effort:*)
            require_env; setup_proxy; uv_sync; run_job "$cmd" job_MMMU_pro_thinking_effort "${cmd#*:}" ;;
        AIME_non_thinking|AIME_non-thinking)
            require_env; setup_proxy; uv_sync; run_job "AIME_non_thinking" job_AIME_non_thinking ;;
        AIME_thinking)
            require_env; setup_proxy; uv_sync; run_job "AIME_thinking" job_AIME_thinking ;;
        toolcall_thinking)
            require_env; setup_proxy; uv_sync; run_job "toolcall_thinking" job_toolcall_thinking ;;
        toolcall_thinking_effort)
            require_env; setup_proxy; uv_sync; run_job "toolcall_thinking_effort:${2:-max}" job_toolcall_thinking_effort "${2:-max}" ;;
        toolcall_thinking_effort:*)
            require_env; setup_proxy; uv_sync; run_job "$cmd" job_toolcall_thinking_effort "${cmd#*:}" ;;
        *)
            echo "ERROR: Unknown job: $cmd" >&2
            list_jobs
            exit 1
            ;;
    esac
}

main "$@"
