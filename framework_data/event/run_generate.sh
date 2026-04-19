#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)

export PYTHONUNBUFFERED=1
: "${OPENAI_API_KEY:?Please set OPENAI_API_KEY before running this script.}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

python "$SCRIPT_DIR/timeline_generator.py" research \
  --model "${MODEL_RESEARCH:-gpt-4o-mini}" \
  --min_events "${MIN_EVENTS_RESEARCH:-5}" \
  --max_events "${MAX_EVENTS_RESEARCH:-8}" \
  --output_dir "${OUTPUT_DIR_RESEARCH:-$SCRIPT_DIR/output}" \
  --max_workers "${MAX_WORKERS_RESEARCH:-1}" \
  --start_from "${START_FROM_RESEARCH:-0}"

python "$SCRIPT_DIR/timeline_generator.py" tutoring \
  --model "${MODEL_TUTORING:-gpt-4o-mini}" \
  --min_events "${MIN_EVENTS_TUTORING:-5}" \
  --max_events "${MAX_EVENTS_TUTORING:-8}" \
  --output_dir "${OUTPUT_DIR_TUTORING:-$SCRIPT_DIR/output}" \
  --max_workers "${MAX_WORKERS_TUTORING:-1}" \
  --start_from "${START_FROM_TUTORING:-0}"