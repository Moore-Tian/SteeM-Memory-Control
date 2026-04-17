
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

python "$SCRIPT_DIR/compose.py" \
  --input_dir "${INPUT_DIR:-$REPO_ROOT/examples/mix/research}" \
  --output_dir "${OUTPUT_DIR:-$SCRIPT_DIR/output}" \
  --episodic_k "${EPISODIC_K:-5}" \
  --max_retrieved "${MAX_RETRIEVED:-8}" \
  --include_anchor_in_episodic \
  --domains "${DOMAINS:-research}"