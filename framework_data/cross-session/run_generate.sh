#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

: "${OPENAI_API_KEY:?Please set OPENAI_API_KEY before running this script.}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_UID="${OPENAI_UID:-}"

python "$SCRIPT_DIR/cross_session_generator.py" \
  --domain research \
  --topics_file "$SCRIPT_DIR/cross_session_topics_research.json" \
  --mix_dir "${MIX_DIR_RESEARCH:-$REPO_ROOT/examples/mix}" \
  --regimes_file "$SCRIPT_DIR/all_pref_regimes.json" \
  --output_dir "${OUTPUT_DIR_RESEARCH:-$SCRIPT_DIR/output}" \
  --model "${MODEL_RESEARCH:-gpt-4o-mini}" \
  --sleep_s "${SLEEP_SECONDS_RESEARCH:-0.0}" \
  --skip_existing \
  --max_workers "${MAX_WORKERS_RESEARCH:-2}" \
  --num_interactions "${NUM_INTERACTIONS_RESEARCH:-5}"

python "$SCRIPT_DIR/cross_session_generator.py" \
  --domain tutoring \
  --topics_file "$SCRIPT_DIR/cross_session_topics_tutoring.json" \
  --mix_dir "${MIX_DIR_TUTORING:-$REPO_ROOT/examples/mix}" \
  --regimes_file "$SCRIPT_DIR/all_pref_regimes.json" \
  --output_dir "${OUTPUT_DIR_TUTORING:-$SCRIPT_DIR/output}" \
  --model "${MODEL_TUTORING:-gpt-4o-mini}" \
  --sleep_s "${SLEEP_SECONDS_TUTORING:-0.0}" \
  --skip_existing \
  --max_workers "${MAX_WORKERS_TUTORING:-2}" \
  --num_interactions "${NUM_INTERACTIONS_TUTORING:-5}"

