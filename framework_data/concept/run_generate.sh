#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

export PYTHONUNBUFFERED=1
: "${OPENAI_API_KEY:?Please set OPENAI_API_KEY before running this script.}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_UID="${OPENAI_UID:-}"

python "$SCRIPT_DIR/concept_generator.py" \
    --input_dir "${INPUT_DIR_RESEARCH:-$REPO_ROOT/examples/mix/research}" \
    --output_dir "${OUTPUT_DIR_RESEARCH:-$SCRIPT_DIR/output}" \
    --model "${MODEL_RESEARCH:-gpt-4o-mini}" \
    --request_type openai \
    --sleep_s "${SLEEP_SECONDS_RESEARCH:-0.0}" \
    --max_workers "${MAX_WORKERS_RESEARCH:-2}" \
    --skip_existing

python "$SCRIPT_DIR/concept_generator.py" \
    --input_dir "${INPUT_DIR_TUTORING:-$REPO_ROOT/examples/mix/tutoring}" \
    --output_dir "${OUTPUT_DIR_TUTORING:-$SCRIPT_DIR/output}" \
    --model "${MODEL_TUTORING:-gpt-4o-mini}" \
    --request_type openai \
    --sleep_s "${SLEEP_SECONDS_TUTORING:-0.0}" \
    --max_workers "${MAX_WORKERS_TUTORING:-2}" \
    --skip_existing