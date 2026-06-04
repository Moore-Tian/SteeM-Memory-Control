#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Set OPENAI_API_KEY before running. OPENAI_BASE_URLis optional.


python artifact_generator.py \
  --input_dir ../timeline_generation/output_gemini/research \
  --output_dir output \
  --model gemini-2.5-pro \
  --max_workers 1 \
  --sleep_s 0.5 \


python artifact_generator.py \
  --input_dir ../timeline_generation/output_gemini/tutoring \
  --output_dir output \
  --model gemini-2.5-pro \
  --max_workers 1 \
  --sleep_s 0.5 \
