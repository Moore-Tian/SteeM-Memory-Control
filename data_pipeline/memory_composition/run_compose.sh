
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python compose.py \
  --input_dir "${INPUT_DIR:?Please set INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR:?Please set OUTPUT_DIR}" \
  --episodic_k 5 \
  --max_retrieved 8 \
  --include_anchor_in_episodic \
  --domains research