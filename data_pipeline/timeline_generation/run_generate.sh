#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Set OPENAI_API_KEY before running. OPENAI_BASE_URL is optional.

# python timeline_generator.py research \
#   --model gemini-2.5-pro \
#   --min_events 1 \
#   --max_events 2 \
#   --output_dir output \
#   --max_workers 1 \
#   --request_type openai \
#   --start_from 100

  python timeline_generator.py tutoring \
  --model gemini-2.5-flash \
  --min_events 5 \
  --max_events 5 \
  --output_dir output \
  --max_workers 1 \
  --request_type openai \
  --start_from 10 \
  --limit 2